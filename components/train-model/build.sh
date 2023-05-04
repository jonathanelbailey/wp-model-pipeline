#! /bin/sh -e

registry=$1
image_name=$2
tag=$3

docker build --no-cache . -t $image_name:$tag
docker image tag data-loader $registry/$image_name:$tag
docker push $registry/$image_name:$tag