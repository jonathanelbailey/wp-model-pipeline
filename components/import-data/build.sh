#! /bin/sh -e

registry=$1
image_name=$2
tag=$3

docker build . -t $image_name:$tag
docker image tag $image_name $registry/$image_name:$tag
docker push $registry/$image_name:$tag