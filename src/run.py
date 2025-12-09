import os
from camel.storages import Neo4jGraph
from dataloader import load_high
import argparse
from creat_graph_with_description import creat_metagraph_with_description
from summerize import process_chunks
from retrieve import seq_ret
from improved_retrieve import get_improved_response, hybrid_retrieve
from utils import *
from nano_graphrag import GraphRAG, QueryParam
from logger_ import get_logger

logger = get_logger("run", log_file="logs/run.log")

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-simple', action='store_true')
parser.add_argument('-construct_graph', action='store_true')
parser.add_argument('-inference',  action='store_true')
parser.add_argument('-improved_inference', action='store_true')
parser.add_argument('-grained_chunk',  action='store_true')
parser.add_argument('-trinity', action='store_true')
parser.add_argument('-trinity_gid1', type=str)
parser.add_argument('-trinity_gid2', type=str)
parser.add_argument('-ingraphmerge',  action='store_true')
parser.add_argument('-crossgraphmerge', action='store_true')
parser.add_argument('-dataset', type=str, default='mimic_ex')
parser.add_argument('-data_path', type=str, default='./dataset_test')
parser.add_argument('-test_data_path', type=str, default='./dataset_ex/report_0.txt')
args = parser.parse_args()

if args.simple:
    graph_func = GraphRAG(working_dir="./nanotest")

    with open("./dataset_ex/report_0.txt") as f:
        graph_func.insert(f.read())

    print(graph_func.query("What is the main symptom of the patient?", param=QueryParam(mode="local")))

else:

    url=os.getenv("NEO4J_URL")
    username=os.getenv("NEO4J_USERNAME")
    password=os.getenv("NEO4J_PASSWORD")

    # Set Neo4j instance
    n4j = Neo4jGraph(
        url=url,
        username=username,
        password=password
    )

    if args.construct_graph: 
        if args.dataset == 'mimic_ex':
            files = [file for file in os.listdir(args.data_path) if os.path.isfile(os.path.join(args.data_path, file))]
            
            for file_name in files:
                file_path = os.path.join(args.data_path, file_name)
                content = load_high(file_path)
                gid = str_uuid()
                n4j = creat_metagraph_with_description(args, content, gid, n4j)

                if args.trinity:
                    link_context(n4j, args.trinity_gid1)
            if args.crossgraphmerge:
                merge_similar_nodes(n4j, None)

    def improved_inference_pipeline(n4j, question: str, use_multi_subgraph: bool = False):
        """
        Improved Inference Pipeline using Hybrid U-Retrieval
        """
        from dedicated_key_manager import create_dedicated_client
        
        logger.info("IMPROVED INFERENCE PIPELINE (Hybrid U-Retrieval)")
        logger.info(f"Question: {question[:200]}...")
        logger.info(f"Multi-subgraph mode: {use_multi_subgraph}")
        
        inference_client = create_dedicated_client(task_id="improved_inference")
        
        logger.info("PHASE 1: HYBRID RETRIEVAL")
        logger.info("[1/4] Vector Search - Pre-filtering candidates...")
        top_k = 3 if use_multi_subgraph else 1
        gids = hybrid_retrieve(n4j, question, inference_client, top_k=top_k)
        
        if not gids:
            logger.error("No relevant subgraphs found!")
            return None
        
        logger.info(f"Selected {len(gids)} subgraph(s): {[g[:8]+'...' for g in gids]}")
        logger.info("PHASE 2: RESPONSE GENERATION")
        
        answer, primary_gid = get_improved_response(
            n4j, 
            question, 
            inference_client,
            use_multi_subgraph=use_multi_subgraph,
            top_k_subgraphs=top_k
        )
        
        if not answer:
            logger.error("Failed to generate answer")
            return None
        
        logger.info("INFERENCE COMPLETE")
        logger.info(f"Primary GID: {primary_gid[:16]}...")
        logger.info(f"Answer length: {len(answer)} characters")
        logger.info(f"Preview: {answer[:300]}...")
        
        return answer

    if args.improved_inference:
        question = load_high("./prompt.txt")
        logger.info(f"Loaded question from prompt.txt")
        
        answer = improved_inference_pipeline(n4j, question, use_multi_subgraph=False)
        
        if answer:
            logger.info("FINAL ANSWER")
            print(answer)
        else:
            logger.error("Cannot perform inference - no valid results found.")
            logger.error("Please ensure the knowledge graph is built and contains relevant data.")

    if args.inference:
        from dedicated_key_manager import create_dedicated_client
        
        question = load_high("./prompt.txt")
        
        inference_client = create_dedicated_client(task_id="inference_main")
        sum = process_chunks(question, client=inference_client)
        gid = seq_ret(n4j, sum)
        
        if gid is None:
            print("\nCannot perform inference - database is empty or no valid results found.")
            print("Run the following command first to build the graph:")
            print("python run.py -dataset mimic_ex -data_path ../data/mimic_ex -construct_graph")
        else:
            response = get_response(n4j, gid, question)
            print(response)
