## VC Client for Docker

[Japanese](/README_ja.md)

## Build

In root folder of repos.

```
npm run build:docker:vcclient
```

## Run

In root folder of repos.

```
bash start_docker.sh
```

Without GPU

```
USE_GPU=off bash start_docker.sh
```

Specify port num

```
EX_PORT=<port> bash start_docker.sh
```

Use Local Image

```
USE_LOCAL=on bash start_docker.sh
```

## Push to Repo (only for devs)

```
npm run push:docker:vcclient
```
