for tag in latest-gpu-py3 latest-gpu-py3
do
    docker build -t matsen/tcr-vae:$tag --build-arg tftag=$tag .
    docker push matsen/tcr-vae:$tag
done
