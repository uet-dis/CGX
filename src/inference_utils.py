# inference_utils.py

from improved_retrieve import get_improved_response, hybrid_retrieve
from logger_ import get_logger

logger = get_logger("inference_utils", log_file="logs/inference_utils.log")

def infer(n4j, question:str, use_multi_subgraph:bool=False):
    from dedicated_key_manager import create_dedicated_client

    logger.info("INFERENCE")
    logger.info(f"Question: {question[:200]}...")
    logger.info(f"Multi-subgraph mode: {use_multi_subgraph}")

    inference_client = create_dedicated_client(task_id="inference")

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