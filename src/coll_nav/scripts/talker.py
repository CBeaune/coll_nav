#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped, Pose
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
import requests
import sys, select, termios, tty

import networkx as nx
import numpy as np

url_vehicles = 'http://127.0.0.1:5000/vehicles'
url_groups = 'http://127.0.0.1:5000/groups'
url_msgs = 'http://127.0.0.1:5000/msgs/vehicles'

#build minimal graph
graph = nx.DiGraph()
edges = [((-1,0),(0,0)),((0,0),(1,0)),((1,0),(2,0)),((1,0),(1,-1)),((1,-1),(2,-1)),((2,-1),(2,0)),((2,-0),(2,1))]
graph.add_edges_from(edges)
waypoints = [(-1,0),(0,0),(1,0),(2,0),(1,-1),(2,-1),(2,1)]

L = 1. #lookahead distance
wp = 0
vx = 0.1
theta = 0.0

x_init = float()
y_init = float()
pose = Pose()
range_left = 100000.0
range_right = 100000.0


def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def to_dict(position, intention, namespace, done=False):
    my_vehicle = dict()
    my_vehicle["id"] = namespace
    my_vehicle["x"] = position.x
    my_vehicle["y"] = position.y
    my_vehicle["intention"] = intention  # str(agent.get_destination()[1])
    my_vehicle["current_state"] = done
    return my_vehicle


def odom_callback(data):
    global pose
    pose = data.pose.pose


def range_r_callback(data):
    global range_right
    range_right = data.range


def range_l_callback(data):
    global range_left
    range_left = data.range


def find_closest_wp(_x, _y):
    x = _x
    y = _y
    dist = 10000
    for wp in waypoints:
        new_dist = (x-wp[0])**2 + (y-wp[1])**2
        if new_dist < dist:
            dist = new_dist
            cl_wp = wp
    return cl_wp


def pure_pursuit(position, waypoints):
    if not waypoints:
        return 0.0, 0.0, 0.0
    if not waypoints[0]:
        return 0.0, 0.0, 0.0
    global L, wp, vx, theta
    X = position.x
    Y = position.y
    x_n0 = waypoints[wp][0]
    y_n0 = waypoints[wp][1]
    
    if wp>len(waypoints)-2:
        x_n1 = waypoints[wp][0]+1.0
        y_n1 = waypoints[wp][1]
    else:
        x_n1 = waypoints[wp+1][0]
        y_n1 = waypoints[wp+1][1]
    up = ((X - x_n0)*(x_n1 - x_n0)+(Y - y_n0)*(y_n1 - y_n0))/((x_n1 - x_n0)**2+(y_n1 - y_n0)**2)
    xo = x_n0 + up*(x_n1 - x_n0)
    yo = y_n0 + up*(y_n1 - y_n0)

    error = np.sqrt((X - xo)**2+(Y - yo)**2)

    if np.sqrt((x_n1 - xo)**2+(y_n1 - yo)**2) <= L:
        if wp<len(waypoints)-2:
            wp=wp+1

    do = np.sqrt((X - xo)**2+(Y - yo)**2)
    if do**2 >= L**2:
        dl = 0
    else:
        dl = np.sqrt(L**2-do**2)

    wpvec = np.array([x_n1 - x_n0, y_n1 - y_n0])
    norm_vec = wpvec/np.linalg.norm(wpvec)

    [xl, yl] = [xo, yo] + dl*norm_vec
    rot_matrix = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    tmp = np.dot(rot_matrix, np.array([[xl-X],[yl-Y]]))
    yd = tmp[1][0]

    w = 2*yd*vx/(L**2)

    return vx, w, error


def get_msg(namespace, my_vehicle):
    # Retrieve data to server
    msgs = requests.get(url_msgs + '/' + namespace)
    msgs = msgs.json()
    for msg in msgs:  # TO DO
        warning_id = msg["warning"]
        group = requests.get(url_groups)
        group = group.json()
        group_range = group[0]
        my_range = group_range[namespace]
        if warning_id in my_range:
            return True
    return False


def node():
    rospy.init_node('talker', anonymous=True)

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    sub = rospy.Subscriber('/odom', Odometry, odom_callback)
    sub = rospy.Subscriber('/range/fl', Range, range_l_callback)
    sub = rospy.Subscriber('/range/fr', Range, range_r_callback)

    # rospy.set_param("~ego", False)

    namespace = rospy.get_param("~namespace")
    threshold = rospy.get_param("~threshold", 0.2)
    x_goal = rospy.get_param("~x_goal", 2)
    y_goal = rospy.get_param("~y_goal", 0)

    # rospy.loginfo(ego)
    rate = rospy.Rate(5)  # 10hz

    my_vehicle = to_dict(pose.position, '(2,0)', namespace)
    r = requests.post(url_vehicles, json=my_vehicle)
    r = requests.patch(url_groups, json=my_vehicle)
    stop_requested = False

    while not rospy.is_shutdown():
        # key = getKey()
        if (range_right < threshold) & (range_left < threshold):
            r = requests.post(url_msgs + '/' + namespace, json=my_vehicle)
            stop_requested = True
            rospy.loginfo(namespace + 'obstacle detected')
        start_wp = find_closest_wp(pose.position.x, pose.position.y)

        affected_by_msgs = get_msg(namespace, my_vehicle)

        if affected_by_msgs:
            rospy.loginfo(namespace + 'received warning msg, replanning')
            if graph.has_edge((1, 0), (2, 0)):
                graph.remove_edge((1, 0), (2, 0))
        path = nx.astar_path(graph, start_wp, (x_goal, y_goal))
        # rospy.loginfo((namespace, pose.position.x, pose.position.y))
        msg = Twist()

        if ((start_wp == (x_goal, 0)) & ((pose.position.x-x_goal) < .5)) | stop_requested:
            rospy.loginfo(namespace + ' stop requested')
            msg.linear.x = 0.0
            msg.angular.z = 0.0
        else:
            vx, w, err = pure_pursuit(pose.position, path)
            msg.linear.x = vx
            msg.angular.z = w
            pub.publish(msg)
            rate.sleep()
        # not ego
        # Send data to server
        my_vehicle = to_dict(pose.position, '(2,0)', namespace)
        r = requests.patch(url_vehicles, json=my_vehicle)
    # Send data to server
    my_vehicle = to_dict(pose.position, '(2,0)', namespace)
    r = requests.delete(url_vehicles, json=my_vehicle)
    r = requests.delete(url_groups, json=my_vehicle)


if __name__ == '__main__':
    try:
        settings = termios.tcgetattr(sys.stdin)
        node()
    except rospy.ROSInterruptException:
        pass
