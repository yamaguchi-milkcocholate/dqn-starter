cd `dirname $0`

if [ -e "config.json" ]
then
    config=$(cat config.json)
    num_devide=$(echo $config | jq '.train_params' | jq '.NUM_DEVIDE')
    echo "K=${num_devide}foldを5コアで実行"
    echo "=============================="
    seq 1 $num_devide | xargs -i python multi.py {}
else
    echo "config.jsonが存在しない"
fi