FROM centos:7

# 更新系统并安装开发工具
RUN yum -y update && \
    yum -y groupinstall "Development Tools" && \
    yum -y install epel-release

# 安装其他依赖项
RUN yum -y install git wget which cmake3
RUN yum -y install libjpeg-turbo-devel libpng-devel libtiff-devel
RUN yum -y install gtk2-devel
RUN yum -y install libdc1394-devel libv4l-devel gstreamer-plugins-base-devel

# 清理缓存
RUN yum -y clean all

# 创建软链接以使用 cmake
RUN ln -s /usr/bin/cmake3 /usr/bin/cmake

# 设置环境变量
ENV PATH="/usr/local/bin:${PATH}"

# 安装 MPICH
RUN wget https://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz && \
    tar -xzf mpich-3.3.2.tar.gz && \
    cd mpich-3.3.2 && \
    ./configure --prefix=/usr/local --includedir=/usr/local/mpi/include && \
    make && make install && \
    cd .. && rm -rf mpich-3.3.2.tar.gz mpich-3.3.2

# 安装 OpenCV 4
# 下载 OpenCV 和 OpenCV Contrib 
RUN set -eux && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    mkdir /opencv-4.x/build
RUN set -eux && \
    cd /opencv-4.x/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-4.x/modules \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D WITH_FFMPEG=NO \
          -D WITH_IPP=NO \
          -D WITH_OPENEXR=NO \
          -D WITH_TBB=YES \
          .. && \
    make -j$(nproc) && make install && \
    cd ../.. && rm -rf opencv.zip opencv_contrib.zip opencv-4.x opencv_contrib-4.x

# 安装cuda
RUN wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-rhel7-11-1-local-11.1.1_455.32.00-1.x86_64.rpm && \
    rpm -i cuda-repo-rhel7-11-1-local-11.1.1_455.32.00-1.x86_64.rpm && \
    yum clean all && \
    yum install -y cuda


# 设置默认命令
CMD ["bash"]

