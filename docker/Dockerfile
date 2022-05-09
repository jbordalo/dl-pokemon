FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04

RUN apt update \
  && apt install -y python3-pip python3-dev \
  && pip3 install --upgrade pip

RUN mkdir -p /work

WORKDIR /work
ADD requirements.txt /work/requirements.txt
RUN pip3 install --no-cache -r requirements.txt

ADD configure-ssh.sh configure-ssh.sh
RUN chmod +x configure-ssh.sh
RUN ./configure-ssh.sh
ENV NOTVISIBLE "in users profile"

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
