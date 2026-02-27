#pragma once
#include <stdlib.h> 
#include <string.h>
#include <stdio.h>
#include "config.h"
typedef struct {
    graph_node_t iter[PAT_SIZE];
    graph_node_t uiter[PAT_SIZE];
    graph_node_t slot_size[MAX_SLOT_NUM][UNROLL];
    graph_node_t (*slot_storage)[UNROLL][GRAPH_DEGREE];
    pattern_node_t level;
} CallStack;

