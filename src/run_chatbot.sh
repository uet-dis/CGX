#!/bin/bash

# Medical Knowledge Graph Chatbot Launcher
# =========================================

echo " Starting Medical Knowledge Graph Chatbot..."
echo ""

# Check if Neo4j environment variables are set
if [ -z "$NEO4J_URL" ] || [ -z "$NEO4J_USERNAME" ] || [ -z "$NEO4J_PASSWORD" ]; then
    echo " Error: Neo4j environment variables not set!"
    echo ""
    echo "Please set the following environment variables:"
    echo "  export NEO4J_URL=your_neo4j_url"
    echo "  export NEO4J_USERNAME=your_username"
    echo "  export NEO4J_PASSWORD=your_password"
    echo ""
    exit 1
fi

echo " Neo4j credentials found"
echo "   URL: $NEO4J_URL"
echo "   Username: $NEO4J_USERNAME"
echo ""

# Change to src directory
cd /home/medgraph/src

echo " Launching Gradio chatbot..."
echo "   - Local URL will be available at http://localhost:7860"
echo "   - Public share link will be generated (valid for 72 hours)"
echo ""
echo "Press Ctrl+C to stop the chatbot"
echo "----------------------------------------"
echo ""

# Run the chatbot
python chatbot_gradio.py
