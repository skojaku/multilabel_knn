apidoc=doc/apis.md
temp_dir=$(mktemp -d)

pdoc emlens -o $temp_dir

cat $temp_dir/emlens/index.md > $apidoc 
for name in $temp_dir/emlens/*.md
do
    if [[ "$name" = $temp_dir"/emlens/index.md" ]]; then
        continue
    fi
    echo "  " |tr ' ' '\n' >>$apidoc 
    cat $name  >>$apidoc
done
rm -r $temp_dir
#rm -r doc/emlens

