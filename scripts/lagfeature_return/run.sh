cd `dirname $0`

if [ -e "config.json" ]
then
    config=$(cat config.json)
    num_divide=$(echo $config | jq '.train_params' | jq '.NUM_DIVIDE')
    echo "K=${num_divide}foldを5コアで実行"
    echo "=============================="
    seq 2 $num_divide | xargs -i python multi.py {}
else
    echo "config.jsonが存在しない"
fi