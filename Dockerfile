# referred: https://zuma-lab.com/posts/docker-python-settings
FROM python:3.9

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
RUN apt-get install -y vim less wget make sudo curl git jq

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN mkdir -p /home/opt
COPY pyproject.toml /home/opt
WORKDIR /home/opt

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install