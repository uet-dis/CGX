import os
from getpass import getpass
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from dataloader import load_high
import argparse
from data_chunk import run_chunk
from utils import *


def creat_metagraph(args, content, gid, n4j):
    """
    Legacy graph creation function (deprecated - use creat_graph_with_description.py instead)
    """
    from dedicated_key_manager import create_dedicated_client

    # Set instance
    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent()
    whole_chunk = content
    
    # Create dedicated client for this graph
    client = create_dedicated_client(task_id=f"legacy_gid_{gid[:8]}")

    if args.grained_chunk == True:
        content = run_chunk(content, client=client)
    else:
        content = [content]
    for cont in content:
        element_example = uio.create_element_from_text(text=cont)

        ans_str = kg_agent.run(element_example, parse_graph_elements=False)
        # print(ans_str)

        graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
        graph_elements = add_ge_emb(graph_elements)
        graph_elements = add_gid(graph_elements, gid)

        n4j.add_graph_elements(graph_elements=[graph_elements])
    if args.ingraphmerge:
        merge_similar_nodes(n4j, gid)
    add_sum(n4j, whole_chunk, gid, client=client)
    return n4j

