graphs=(
    https://snap.stanford.edu/data/email-Enron.txt.gz
    #https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz
    #https://github.com/HPC-Research-Lab/STMatch/blob/main/data/local_txt_graph/mico.txt
    #https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz

)
txtdir=../data/txt_graph/
bindir=../data/bin_graph/
mkdir -p ${txtdir}
mkdir -p ${bindir}

for graph in ${graphs[@]}
do
    wget -nc -P ${txtdir} ${graph}
done
gzip -f -d ${txtdir}*

# cp ../data/local_txt_graph/mico.txt ${txtdir}

for graph in ${txtdir}*
do
    filename=${graph##*/}
    filename=${filename%.txt}
    mkdir -p ${bindir}/${filename}
    cp ${graph} ${bindir}/${filename}/snap.txt
done

for graph in ${bindir}*
do
    bash convert.sh ${graph}
done

cd ..
sed -i '/Graph\* to_gpu() {/,/}/ s/^/\/\//' src/graph.h
sed -i '/Pattern\* to_gpu() {/,/}/ s/^/\/\//' src/pattern.h
g++ key_nodes.cpp -o key_nodes
./key_nodes email-Enron
# ./key_nodes com-youtube.ungraph
# ./key_nodes mico
# ./key_nodes com-orkut.ungraph

sed -i '/Graph\* to_gpu() {/,/}/ s/^\/\///' src/graph.h
sed -i '/Pattern\* to_gpu() {/,/}/ s/^\/\///' src/pattern.h