if [ "$#" -eq 0 ]
then
	uv run -m training.train 2> error.txt
elif [ "$#" -eq 1 ]
then
	uv run -m training.train $1 2> error.txt
else
	echo "ERR: Too many arguments passed"
	exit 1
fi
