#!/bin/bash
DEBIAN_FRONTEND=noninteractive
TZ=Europe/Lisbon
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
apt-get update
apt-get install -y ssh openssh-server
mkdir /var/run/sshd
#Change your password here:
echo 'root:kurwa' | chpasswd
# Root password was changed with testssh
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
echo "export VISIBLE=now" >> /etc/profile