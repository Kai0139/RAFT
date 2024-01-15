FROM ubuntu:focal
ARG ubuntu_release=focal

RUN groupadd -r user && useradd -r -g user user --create-home
# Install ROS Noetic
USER root
RUN apt update
# USER user
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $ubuntu_release main" > /etc/apt/sources.list.d/ros-latest.list'
# USER root
RUN apt-get install curl gnupg gnupg1 gnupg2 -y
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update
RUN DEBIAN_FRONTEND='noninteractive' apt install ros-noetic-desktop-full -y
RUN apt install python3-rosdep python3-rosinstall python3-rosinstall-generator \
    python3-wstool build-essential python3-pip python3-rosdep -y
RUN rosdep init && rosdep update
# Setup environment
# USER user
# RUN cd $HOME && mkdir -p catkin_ws/src
# RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
# RUN /bin/bash -c "cd $HOME/catkin_ws/ &&\
#     source $HOME/.bashrc && source /opt/ros/noetic/setup.bash &&\
#     catkin_make"
RUN pip3 install scipy packaging tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
# For cuda 11.7: pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD ["bash"]