#!/bin/bash

# 定义图名列表
GRAPH_NAMES="sk-2005 uk-2005 twitter-2010"

# 遍历每个图名
for GRAPH_NAME in $GRAPH_NAMES; do
    # 为图名添加 dataset 目录前缀
    FULL_GRAPH_NAME="dataset/$GRAPH_NAME"
    
    # 执行 Java 命令
    java -classpath .:webgraph-3.6.10.jar:dsiutils-2.6.17.jar:fastutil-8.5.5.jar:log4j-1.2.17.jar:slf4j-simple-2.0.0-alpha3.jar:slf4j-api-2.0.0-alpha3.jar:jsap-2.0a.jar BV2Ascii $FULL_GRAPH_NAME > $FULL_GRAPH_NAME.txt
    
    # 输出执行信息
    echo "已将 $FULL_GRAPH_NAME 的输出保存到 $FULL_GRAPH_NAME.txt"
done