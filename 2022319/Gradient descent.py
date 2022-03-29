#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/24 21:44
# @Author  : Cheng liwei
# @FileName: Gradient descent.py
# @Software: PyCharm

import torch
import numpy as np


#当前x值的更新是新的x等于上一个x减去（函数的导数值（即梯度）*学习速率）
## len() 的用法 返回对象的数目
def compute_error_for_line_given_points(b,w,points):
    totalError=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        totalError += (y-(w*x+b))**2
    return totalError / float(len(points))

def step_gradient(b_current,w_current,points,learningRate):
    b_gradient=0
    w_gradient=0
    N=float(len(points))

    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]

        b_gradient += -(2/N)*(y-(w_current*x+b_current))
        w_gradient += -(2/N) * x * (y-((w_current*x)+b_current))

    new_b=b_current-(learningRate*b_gradient)
    new_m=w_current-(learningRate*w_gradient)

    return  [new_b,new_m]

def gradient_descent_runner(points,starting_b,starting_m,learningRate,num_iterations):
    b=starting_b
    m=starting_m

    for i in range(num_iterations):
        b,m=step_gradient(b,m,np.array(points),learningRate)
    return [b, m]

#通过genfromtxt 读取数据 format函数的用法是代替 str.format用于代替前文字符串的%  即{0} {1}

def run():
    points=np.genfromtxt("data.csv",delimiter=",")
    learningRate=0.0001
    initial_b=0
    initial_a=0
    num_iterations=1000

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".
          format(initial_b,initial_a,
                 compute_error_for_line_given_points(initial_b,initial_a,points)))

    print("delay")
    [b,m]=gradient_descent_runner(points, initial_b, initial_a, learningRate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, m,
                 compute_error_for_line_given_points(b, m, points))
          )

if __name__ == '__main__':
    run()




