"""
app.py

FastAPI backend server for video transcription and analysis pipeline:
1. Provides REST API endpoints for video processing
2. WebSocket support for real-time progress updates
3. Serves static frontend files

Requirements:
- Azure Speech Service for transcription
- Azure OpenAI Service for selling points extraction
- Azure Content Understanding for video analysis
- ffmpeg for audio extraction
"""

import os
import glob
import logging
import json
import time
import subprocess
from pathlib import Path
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import argparse
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from openai import AzureOpenAI

# Import visualization library
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import the transcription functions from our module
from transcribe_videos import (
    extract_audio_from_video,
    transcribe_audio_with_word_timestamps,
    transcribe_audio_with_sentence_timestamps
)
from content_understanding_client import AzureContentUnderstandingClient

def setup_logging(log_level='DEBUG'):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper(), logging.DEBUG)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s [%(name)s]: %(message)s')

    # Enable debug logging for HTTP requests only if DEBUG level
    if level == logging.DEBUG:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
        logging.getLogger("requests").setLevel(logging.DEBUG)
        logging.getLogger("azure").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
    else:
        # Set higher levels for noisy loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("azure").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)

# Initial setup with INFO level (will be reconfigured based on args)
setup_logging('INFO')

# Load environment variables
load_dotenv(override=True)
SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')
SPEECH_ENDPOINT = os.getenv('AZURE_SPEECH_ENDPOINT')
OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')
CONTENT_UNDERSTANDING_ENDPOINT = os.getenv('AZURE_CONTENT_UNDERSTANDING_ENDPOINT')
CONTENT_UNDERSTANDING_API_VERSION = os.getenv('AZURE_CONTENT_UNDERSTANDING_API_VERSION')
CONTENT_UNDERSTANDING_API_KEY = os.getenv('AZURE_CONTENT_UNDERSTANDING_API_KEY')

# Verify environment variables
required_vars = {
    'AZURE_SPEECH_KEY': SPEECH_KEY,
    'AZURE_SPEECH_ENDPOINT': SPEECH_ENDPOINT,
    'AZURE_OPENAI_API_KEY': OPENAI_API_KEY,
    'AZURE_OPENAI_API_VERSION': OPENAI_API_VERSION,
    'AZURE_OPENAI_ENDPOINT': OPENAI_ENDPOINT,
    'AZURE_OPENAI_DEPLOYMENT': OPENAI_DEPLOYMENT,
    'AZURE_CONTENT_UNDERSTANDING_ENDPOINT': CONTENT_UNDERSTANDING_ENDPOINT,
    'AZURE_CONTENT_UNDERSTANDING_API_VERSION': CONTENT_UNDERSTANDING_API_VERSION
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    exit(1)

# Initialize FastAPI app
app = FastAPI(title="Video Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Processing status storage
processing_status: Dict[str, Dict[str, Any]] = {}

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=3)

# Pydantic models
class ProcessVideoRequest(BaseModel):
    video_name: str

class BatchProcessRequest(BaseModel):
    video_names: list[str]
    enable_content_understanding: bool = True

class VideoInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    duration: Optional[float] = None
    thumbnail_url: Optional[str] = None

def extract_selling_points(transcription_text):
    """
    Use Azure OpenAI to extract selling points from the transcription text.
    
    Args:
        transcription_text (str): The sentence-level transcription text
        
    Returns:
        list: A list of extracted selling points
    """
    # Handle empty or None transcription
    if not transcription_text or not transcription_text.strip():
        logging.warning("Empty or None transcription provided to extract_selling_points")
        return []
    
    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,  # Update this as needed
            azure_endpoint=OPENAI_ENDPOINT
        )

        prompt = f"""
        {transcription_text}
        """
        
        # Call the Azure OpenAI service
        # Dynamically set max_tokens based on the length of the transcription_text
        # Rough heuristic: 1 token ≈ 4 characters in English
        approx_tokens = max(256, min(len(transcription_text) // 4, 4096))

        response = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
            messages=[
            # {"role": "system", "content": "You are an AI assistant that extracts selling points from transcriptions."},
            {"role": "system", "content": """
Your task is to analyze video transcript and extract the unique and individual selling points mentioned in the transcript. Only list the selling points, no other explanation need to be provided. Pur each selling point as a separate item in a JSON array.
Make sure the selling points word is exactly the same as they appear in the transcript. Long transcript sentences can be broken down into multiple selling points.
 
Here is a list of sample selling point for your reference:

Selling Points list:
Magical pockets set me free!
They come in multiple colors. 
So soft and super stretchy
Get dressed in effortless fashion!
Built-in shorts
Adjustable drawstrings
Stretchy & crazy comfortable!
Stretchy fabric
Built-in shorts provides
Designed straps
Fleece-lining to keep you cosy
Crossover waist design!
Built-in shorts with side pockets
Built-in shorts for easy coverage
Removable pads for customized support
Breathable material for hot days
4-way stretch for easy movement
Pullover hood gives easy coverage
Kangaroo pocket for accessible storage
Move freely without any discomfort
Doesn't rub against my skin
the fabric is soooo stretchy
Duper stretchy for easy movement
Inner lining for added coverage
100% sweat proof.
Easily pat it off
Super soft, super breathable. 
flattering shape and fit
Shows off my curves
So effortless, so elegant!
The comfiest built-in shorts! 
coverage for your underarm
backless and twist design
Yes, 100% squat proof.
Perfect for working out
Pockets to store items
comfortable to the touch
Deep pocket
So many fun colors
Soft and super stretchy
Perfect for everyday wear
The fabric is perfect 
Breathable and sweat-wicking
Basic wardrobe staple
Soft and stretchy 
Classic curved design
Slight flare design
Buttery soft fabric 
Roomy pockets! 
Teardrop back design 
comfortable double straps
UNATTRACTIVE SHAPE? GONE"
Hourglass bodyshape effect
INELASTIC FABRIC? GONE
Back waistband pocket
Round neck cut-out 
Adjustable shoulder straps
Side slit drawstring 
Multi-layer skirt design
Front slit design
Tie-back Backless design
Highlights your curves
Will not shrink 
Tight crotch jeans
Removable cups
Itchy skin
Twist design
Great value
Multi-layered design
Baggy knees
High-waisted band 
Adjustable straps 
Flattering silhouette
Drawstring design
U-shape neckline
Decorative straps 
U-neck racerback
 
New selling points may be mentioned in the transcript that are not included in the list above. In this case, use your best judgement.
             """},
            {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            # max_tokens=approx_tokens,
            max_tokens=200,
            top_p=1,
            response_format={"type": "json_object"}
        )

        # Parse the response
        content = response.choices[0].message.content
        print(content)
        selling_points = json.loads(content).get("selling_points", [])
        
        return selling_points
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error("Error extracting selling points: %s", str(e), extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "transcription_length": len(transcription_text) if transcription_text else 0
        })
        logging.debug("Full error traceback for selling points extraction:\n%s", error_details)
        return []

def match_selling_points_with_timestamps(word_segments, selling_points):
    """
    Match selling points with word-level timestamps.
    
    Args:
        word_segments (list): List of tuples (start_time, end_time, word)
        selling_points (list): List of selling point strings
        
    Returns:
        list: Selling points with timestamp information
    """
    result = []
    
    # Create a lowercase version of word_segments for case-insensitive matching
    word_data = [(start, end, word.lower()) for start, end, word in word_segments]
    
    # Track which words have already been matched
    matched_positions = [False] * len(word_data)
    
    # Process selling points in order (we'll sort by timestamp at the end)
    for selling_point in selling_points:
        # Split the selling point into individual words and convert to lowercase
        point_words = selling_point.lower().split()
        
        # Skip empty selling points
        if not point_words:
            continue
            
        start_time = None
        end_time = None
        matched_indices = []
        
        # First pass: try to find a match using only unmatched words
        for i in range(len(word_data)):
            # Skip if this position is already matched or if there aren't enough words left
            if matched_positions[i] or i + len(point_words) > len(word_data):
                continue
                
            matched_words = 0
            curr_indices = []
            
            for j in range(len(point_words)):
                if i + j >= len(word_data) or matched_positions[i + j]:
                    break
                
                # Check if the current word in the transcription matches the current word in the selling point
                if word_data[i + j][2] in point_words[j] or point_words[j] in word_data[i + j][2]:
                    matched_words += 1
                    curr_indices.append(i + j)
                else:
                    break
            
            # If we have a match for all words or a significant portion
            if matched_words >= max(1, len(point_words) // 2):
                # Set start time from the first matched word
                start_time = word_data[i][0]
                
                # Update end time with the last matched word
                end_time = word_data[i + matched_words - 1][1]
                matched_indices = curr_indices
                
                # For longer selling points, try to find matches for remaining words
                remaining_point_words = " ".join(point_words[matched_words:])
                if remaining_point_words:
                    for k in range(i + matched_words, len(word_data)):
                        if matched_positions[k]:
                            continue
                            
                        if word_data[k][2] in remaining_point_words or any(pw in word_data[k][2] for pw in point_words[matched_words:]):
                            end_time = word_data[k][1]
                            matched_indices.append(k)
                
                # We found a match, so break out of the loop
                break
        
        # Second pass: if no match found in unmatched regions, try anywhere
        if start_time is None:
            for i in range(len(word_data)):
                # Skip if there aren't enough words left
                if i + len(point_words) > len(word_data):
                    continue
                    
                matched_words = 0
                curr_indices = []
                
                for j in range(len(point_words)):
                    if i + j >= len(word_data):
                        break
                    
                    # Check if current word matches
                    if word_data[i + j][2] in point_words[j] or point_words[j] in word_data[i + j][2]:
                        matched_words += 1
                        curr_indices.append(i + j)
                    else:
                        break
                
                # If we have a match
                if matched_words >= max(1, len(point_words) // 2):
                    # Set start time from the first matched word
                    start_time = word_data[i][0]
                    
                    # Update end time with the last matched word
                    end_time = word_data[i + matched_words - 1][1]
                    matched_indices = curr_indices
                    
                    # For longer selling points, try to find matches for remaining words
                    remaining_point_words = " ".join(point_words[matched_words:])
                    if remaining_point_words:
                        for k in range(i + matched_words, len(word_data)):
                            if word_data[k][2] in remaining_point_words or any(pw in word_data[k][2] for pw in point_words[matched_words:]):
                                end_time = word_data[k][1]
                                matched_indices.append(k)
                    
                    break
        
        # If we found timestamps, add to results and mark words as matched
        if start_time is not None and end_time is not None:
            result.append({
                "startTime": round(start_time, 2),
                "endTime": round(end_time, 2),
                "content": selling_point
            })
            
            # Mark the matched positions as used
            for idx in matched_indices:
                matched_positions[idx] = True
        else:
            # If no match was found, include the selling point without timestamps
            result.append({
                "startTime": None,
                "endTime": None,
                "content": selling_point
            })
    
    # Sort results by start time (None values at the end)
    result.sort(key=lambda x: (x["startTime"] is None, x["startTime"]))
    
    return result

def merge_segments_by_selling_points(content_json, selling_points_json, time_deviation_ms=0, min_overlap_percentage=0.2):
    """
    Merge video segments based on selling points timestamps with optional time deviation
    
    Args:
        content_json: Content understanding output JSON
        selling_points_json: Selling points with timestamps JSON
        time_deviation_ms: Time deviation in milliseconds to allow for overlap matching (default: 0ms)
        min_overlap_percentage: Minimum percentage of overlap required (0.0-1.0) to consider a match (default: 0.2)
    
    Returns:
        Dictionary with merged segments, final segments and unmerged content
    """
    result = {
        "merged_segments": [],
        "unmerged_segments": [],
        "final_segments": []
    }
    
    # Get video segments from content understanding output
    # The prebuilt-videoAnalyzer format has segments nested in contents[0].segments
    if "contents" in content_json["result"] and content_json["result"]["contents"]:
        video_segments = content_json["result"]["contents"][0].get("segments", [])
    else:
        video_segments = []
    
    # Track which segments have been merged
    merged_segment_indices = set()
    
    # Process each selling point
    for selling_point in selling_points_json["selling_points"]:
        # Skip if no timestamps
        if selling_point["startTime"] is None or selling_point["endTime"] is None:
            # Add to result as a selling point without timestamp information
            result["merged_segments"].append({
                "startTimeMs": None,
                "endTimeMs": None,
                "content": selling_point["content"],
                "overlapping_segments": []
            })
            logging.info(f"Skipping timing match for selling point without timestamps: {selling_point['content']}")
            continue
        
        # Convert to milliseconds for comparison
        start_time_ms = int(selling_point["startTime"] * 1000)
        end_time_ms = int(selling_point["endTime"] * 1000)
        selling_point_duration = end_time_ms - start_time_ms
        
        # Find overlapping segments with time deviation
        overlapping_segments = []
        for i, segment in enumerate(video_segments):
            # Check if segments overlap with time deviation
            segment_start = segment["startTimeMs"]
            segment_end = segment["endTimeMs"]
            segment_duration = segment_end - segment_start
            
            # Calculate overlap
            overlap_start = max(segment_start, start_time_ms - time_deviation_ms)
            overlap_end = min(segment_end, end_time_ms + time_deviation_ms)
            
            if overlap_start < overlap_end:  # There is some overlap
                # Calculate overlap duration
                overlap_duration = overlap_end - overlap_start
                
                # Calculate overlap percentage relative to the shorter duration
                shorter_duration = min(segment_duration, selling_point_duration)
                overlap_percentage = overlap_duration / shorter_duration
                
                # Only consider as overlapping if percentage is above threshold
                if overlap_percentage >= min_overlap_percentage:
                    overlapping_segments.append({
                        "startTimeMs": segment["startTimeMs"],
                        "endTimeMs": segment["endTimeMs"],
                        "sellingPoint": "",  # prebuilt-videoAnalyzer doesn't have sellingPoint field
                        "description": segment.get("description", "")
                    })
                    # Mark this segment as merged
                    merged_segment_indices.add(i)
                    logging.info(f"Merged segment {i} with selling point '{selling_point['content']}', overlap: {overlap_percentage:.2f}")
                else:
                    logging.info(f"Skipped merging segment {i} with selling point '{selling_point['content']}', insufficient overlap: {overlap_percentage:.2f}")
        
        # Create merged segment
        merged_segment = {
            "startTimeMs": start_time_ms,
            "endTimeMs": end_time_ms,
            "content": selling_point["content"],
            "overlapping_segments": overlapping_segments
        }
        
        result["merged_segments"].append(merged_segment)
    # Only include segments that weren't merged in the unmerged_segments list
    for i, segment in enumerate(video_segments):
        if i not in merged_segment_indices:
            result["unmerged_segments"].append({
                "startTimeMs": segment["startTimeMs"],
                "endTimeMs": segment["endTimeMs"],
                "sellingPoint": "",  # prebuilt-videoAnalyzer doesn't have sellingPoint field
                "description": segment.get("description", "")
            })
    
    # Create final segments from merged segments with overlapping segments
    for merged_segment in result["merged_segments"]:
        if merged_segment["overlapping_segments"]:
            # Get startTimeMs from first overlapping segment
            start_time_ms = merged_segment["overlapping_segments"][0]["startTimeMs"]
            
            # Get endTimeMs from last overlapping segment
            end_time_ms = merged_segment["overlapping_segments"][-1]["endTimeMs"]
            
            # Create final segment
            final_segment = {
                "startTimeMs": start_time_ms,
                "endTimeMs": end_time_ms,
                "sellingPoint": merged_segment["content"]
            }
            
            result["final_segments"].append(final_segment)
    
    # Add all unmerged segments to final_segments as well
    for unmerged_segment in result["unmerged_segments"]:
        final_segment = {
            "startTimeMs": unmerged_segment["startTimeMs"],
            "endTimeMs": unmerged_segment["endTimeMs"],
            "sellingPoint": unmerged_segment["sellingPoint"]
        }
        result["final_segments"].append(final_segment)
    
    return result

def list_available_analyzers(endpoint: str, api_version: str, api_key: str) -> list:
    """
    List all available analyzers to find valid baseAnalyzerId values
    """
    try:
        cu_client = AzureContentUnderstandingClient(
            endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
            x_ms_useragent="azure-ai-content-understanding-python/video_analysis",
        )
        
        analyzers = cu_client.get_all_analyzers()
        logging.debug("Available analyzers: %s", json.dumps(analyzers, indent=2))
        return analyzers
    except Exception as e:
        logging.error("Failed to list analyzers: %s", str(e))
        return []

def analyze_video(video_path: str, 
                 endpoint: str, 
                 api_version: str,
                 analyzer_template_path: str,
                 timeout_seconds: int = 3600,
                 delete_analyzer_after: bool = True) -> Optional[Path]:
    """
    Analyzes a video using Azure Content Understanding client and saves results to a JSON file.
    
    Args:
        video_path: Path to the video file
        endpoint: Azure AI service endpoint
        api_version: Azure AI service API version
        analyzer_template_path: Path to the analyzer template JSON
        timeout_seconds: Timeout for analysis completion in seconds
        delete_analyzer_after: Whether to delete the analyzer after analysis
        
    Returns:
        Path to the output JSON file or None if analysis failed
    """
    try:
        # Create path objects
        video_file = Path(video_path)
        output_json = Path(f"{video_file}.json")
        
        # Set up Azure credentials
        # credential = DefaultAzureCredential()
        # token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        
        # Generate unique analyzer ID
        analyzer_id = f"video_scene_chapter_{uuid.uuid4()}"
        
        # Create Content Understanding client
        cu_client = AzureContentUnderstandingClient(
            endpoint=endpoint,
            api_version=api_version,
            api_key=CONTENT_UNDERSTANDING_API_KEY,
            x_ms_useragent="azure-ai-content-understanding-python/video_analysis",
        )
        
        # Create analyzer
        logging.info("Creating analyzer with ID: %s", analyzer_id, extra={"analyzer_id": analyzer_id})
        response = cu_client.begin_create_analyzer(analyzer_id, analyzer_template_path=analyzer_template_path)
        result = cu_client.poll_result(response)
        
        # Submit video for analysis
        logging.info("Submitting video for analysis: %s", video_file.name, extra={"video": video_file.name})
        response = cu_client.begin_analyze(analyzer_id, file_location=str(video_file))
        
        # Wait for analysis to complete
        video_cu_result = cu_client.poll_result(response, timeout_seconds=timeout_seconds)
        
        # Save results to JSON file
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(video_cu_result, f, indent=2)
        
        logging.info("Analysis complete. Results saved to: %s", output_json, extra={"output_file": str(output_json)})
        
        # Delete analyzer if requested
        if delete_analyzer_after:
            cu_client.delete_analyzer(analyzer_id)
            logging.info("Analyzer deleted: %s", analyzer_id, extra={"analyzer_id": analyzer_id})
        
        return output_json
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error("Video analysis failed for %s", video_path, extra={
            "video": video_path,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_details
        })
        logging.debug("Full error traceback:\n%s", error_details)
        return None

async def update_status(video_name: str, status: str, progress: int, message: str = ""):
    """Update processing status and broadcast via WebSocket"""
    processing_status[video_name] = {
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast({
        "type": "status_update",
        "video_name": video_name,
        "status": status,
        "progress": progress,
        "message": message
    })

async def process_video_async(video_path: str, video_name: str):
    """
    Async wrapper for video processing with status updates
    """
    try:
        await update_status(video_name, "processing", 0, "Starting video processing...")
        
        base = os.path.splitext(video_path)[0]
        word_txt_path = base + "_word.txt"
        sentence_txt_path = base + "_sentence.txt"
        selling_points_path = base + "_selling_points.json"
        merged_segments_path = base + "_merged_segments.json"
        visualization_path = base + "_segments_visualization.png"
        content_json_path = video_path + ".json"
        audio_path = base + ".wav"
        
        # Step 1: Content Understanding Analysis (always enabled)
        await update_status(video_name, "processing", 10, "Analyzing video content...")
        analyzer_template_path = "./analyzer_templates/video_content_understanding.json"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor, 
            analyze_video, 
            video_path, 
            CONTENT_UNDERSTANDING_ENDPOINT, 
            CONTENT_UNDERSTANDING_API_VERSION, 
            analyzer_template_path
        )
        
        # Step 2: Extract audio
        await update_status(video_name, "processing", 30, "Extracting audio from video...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, extract_audio_from_video, video_path, audio_path)
        
        # Step 3: Word-level transcription
        await update_status(video_name, "processing", 40, "Transcribing audio (word-level)...")
        word_segments = await loop.run_in_executor(
            executor, 
            transcribe_audio_with_word_timestamps, 
            audio_path, 
            SPEECH_KEY, 
            SPEECH_ENDPOINT
        )
        
        with open(word_txt_path, "w", encoding="utf-8") as f:
            for start, end, word in word_segments:
                f.write(f"[{start:.2f} - {end:.2f}] {word}\n")
        
        # Step 4: Sentence-level transcription
        await update_status(video_name, "processing", 50, "Transcribing audio (sentence-level)...")
        sentence_segments = await loop.run_in_executor(
            executor,
            transcribe_audio_with_sentence_timestamps,
            audio_path,
            SPEECH_KEY,
            SPEECH_ENDPOINT
        )
        
        with open(sentence_txt_path, "w", encoding="utf-8") as f:
            for start, end, sentence in sentence_segments:
                f.write(f"[{start:.2f} - {end:.2f}] {sentence}\n")
        
        # Step 5: Extract selling points
        await update_status(video_name, "processing", 70, "Extracting selling points...")
        transcription_text = "\n".join([sentence for _, _, sentence in sentence_segments])
        selling_points = await loop.run_in_executor(executor, extract_selling_points, transcription_text)
        
        # Step 6: Match selling points with timestamps
        await update_status(video_name, "processing", 80, "Matching selling points with timestamps...")
        timestamped_selling_points = match_selling_points_with_timestamps(word_segments, selling_points)
        
        with open(selling_points_path, "w", encoding="utf-8") as f:
            json.dump({"selling_points": timestamped_selling_points}, f, indent=2)
        
        # Step 7: Merge segments (content analysis is always done)
        if os.path.exists(content_json_path):
            await update_status(video_name, "processing", 90, "Merging video segments...")
            
            with open(content_json_path, 'r') as f:
                content_json = json.load(f)
            
            with open(selling_points_path, 'r') as f:
                selling_points_json = json.load(f)
            
            merged_segments = merge_segments_by_selling_points(
                content_json, 
                selling_points_json,
                time_deviation_ms=0,
                min_overlap_percentage=0.2
            )
            
            with open(merged_segments_path, 'w') as f:
                json.dump(merged_segments, f, indent=2)
            
            # Step 8: Generate visualization
            await update_status(video_name, "processing", 95, "Generating visualization...")
            await loop.run_in_executor(
                executor,
                create_segments_visualization,
                merged_segments_path,
                visualization_path
            )
        
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        await update_status(video_name, "completed", 100, "Processing completed successfully!")
        
    except Exception as e:
        logging.error(f"Error processing video {video_name}: {str(e)}")
        await update_status(video_name, "error", 0, f"Error: {str(e)}")

def create_segments_visualization(merged_segments_path: str, output_path: str) -> None:
    """
    Create a visualization of video segments showing merged and unmerged segments.
    
    Args:
        merged_segments_path: Path to the merged segments JSON file
        output_path: Path where the visualization PNG will be saved
    """
    try:
        # Load merged segments data
        with open(merged_segments_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract segments
        merged_segments = data.get('merged_segments', [])
        unmerged_segments = data.get('unmerged_segments', [])
        final_segments = data.get('final_segments', [])
        
        # Find max time for x-axis
        max_time = 0
        for seg in merged_segments + unmerged_segments + final_segments:
            if seg.get('endTimeMs'):
                max_time = max(max_time, seg['endTimeMs'])
        
        # Also check overlapping segments
        for seg in merged_segments:
            for overlap in seg.get('overlapping_segments', []):
                if overlap.get('endTimeMs'):
                    max_time = max(max_time, overlap['endTimeMs'])
        
        # Convert to seconds
        max_time_seconds = max_time / 1000
        
        # Reset matplotlib style and use clean professional style
        plt.style.use('default')
        
        # Determine which segments types exist
        has_unmerged = len(unmerged_segments) > 0
        has_final = len(final_segments) > 0
        
        # Dynamically adjust figure height and y-positions based on content
        if has_unmerged:
            fig_height = 10
            y_positions = {
                'overlapping': 4,
                'merged': 3.2, 
                'unmerged': 2.2, 
                'final': 1.2
            }
            y_labels = ['Final Segments', 'Unmerged Segments', 'Selling Point Segments', 'Original Segments']
            y_ticks = [1.2, 2.2, 3.2, 4]
            y_lim = (0.8, 4.5)
        else:
            fig_height = 8
            y_positions = {
                'overlapping': 3.2,
                'merged': 2.4, 
                'final': 1.2
            }
            y_labels = ['Final Segments', 'Selling Point Segments', 'Original Segments']
            y_ticks = [1.2, 2.4, 3.2]
            y_lim = (0.8, 3.7)
        
        # Create figure and axis with Fluent UI colors
        fig, ax = plt.subplots(figsize=(16, fig_height))
        fig.patch.set_facecolor('#f3f2f1')  # Fluent neutral-background-2
        ax.set_facecolor('#ffffff')         # Fluent neutral-background-1
        
        # Define Fluent UI color palette
        brand_primary = '#0078d4'       # Microsoft brand blue
        brand_secondary = '#605e5c'     # Neutral foreground secondary
        merged_color = '#0078d4'        # Brand primary
        overlap_color = '#d13438'       # Danger foreground
        unmerged_color = '#8661c5'      # Purple variant
        final_color = '#107c10'         # Dark green
        text_color = '#242424'          # Neutral foreground 1
        grid_color = '#e1dfdd'          # Neutral stroke 2
        border_color = '#d1d1d1'        # Neutral stroke 1
        
        # Height settings - all segments same height
        bar_height = 0.5
        
        # Plot merged segments
        for seg in merged_segments:
            if seg.get('startTimeMs') is not None and seg.get('endTimeMs') is not None:
                start = seg['startTimeMs'] / 1000
                duration = (seg['endTimeMs'] - seg['startTimeMs']) / 1000
                
                # Draw merged segment with clean styling
                rect = patches.Rectangle(
                    (start, y_positions['merged'] - bar_height/2),
                    duration, bar_height,
                    linewidth=1, edgecolor=border_color, facecolor=merged_color,
                    alpha=0.9
                )
                ax.add_patch(rect)
                
                # Add label with Fluent UI styling
                content = seg.get('content', '')
                if len(content) > 35:
                    content = content[:32] + '...'
                if content and duration > 0.5:  # Only show text if segment is wide enough
                    ax.text(start + duration/2, y_positions['merged'], content,
                           ha='center', va='center', fontsize=9, color='white',
                           weight='500', family='Segoe UI',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=merged_color, alpha=0.95, 
                                   edgecolor='none'))
                
                # Plot overlapping segments
                for i, overlap in enumerate(seg.get('overlapping_segments', [])):
                    if overlap.get('startTimeMs') is not None and overlap.get('endTimeMs') is not None:
                        overlap_start = overlap['startTimeMs'] / 1000
                        overlap_duration = (overlap['endTimeMs'] - overlap['startTimeMs']) / 1000
                        
                        # Draw overlapping segment
                        overlap_rect = patches.Rectangle(
                            (overlap_start, y_positions['overlapping'] - bar_height/2),
                            overlap_duration, bar_height,
                            linewidth=1, edgecolor=border_color, facecolor=overlap_color,
                            alpha=0.8
                        )
                        ax.add_patch(overlap_rect)
                        
                        # Add segment label if duration is sufficient
                        if overlap_duration > 0.4:
                            overlap_text = overlap.get('sellingPoint', f'Overlap {i+1}')
                            if len(overlap_text) > 12:
                                overlap_text = overlap_text[:9] + '...'
                            ax.text(overlap_start + overlap_duration/2, y_positions['overlapping'], 
                                   overlap_text, ha='center', va='center', fontsize=8, 
                                   color='white', weight='500', family='Segoe UI')
                        
                        # Add arrow from overlapping segment to merged segment
                        arrow_start_x = overlap_start + overlap_duration/2
                        arrow_start_y = y_positions['overlapping'] - bar_height/2
                        arrow_end_x = start + duration/2
                        arrow_end_y = y_positions['merged'] + bar_height/2
                        
                        # Draw arrow using annotate with straight line
                        ax.annotate('', xy=(arrow_end_x, arrow_end_y), 
                                   xytext=(arrow_start_x, arrow_start_y),
                                   arrowprops=dict(arrowstyle='->', color=brand_secondary, 
                                                 alpha=0.6, linewidth=1.5))
        
        # Plot unmerged segments (only if they exist)
        if has_unmerged:
            for seg in unmerged_segments:
                if seg.get('startTimeMs') is not None and seg.get('endTimeMs') is not None:
                    start = seg['startTimeMs'] / 1000
                    duration = (seg['endTimeMs'] - seg['startTimeMs']) / 1000
                    
                    rect = patches.Rectangle(
                        (start, y_positions['unmerged'] - bar_height/2),
                        duration, bar_height,
                        linewidth=1, edgecolor=border_color, facecolor=unmerged_color,
                        alpha=0.8
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    selling_point = seg.get('sellingPoint', '')
                    if selling_point and len(selling_point) > 25:
                        selling_point = selling_point[:22] + '...'
                    if selling_point and duration > 0.5:
                        ax.text(start + duration/2, y_positions['unmerged'], selling_point,
                               ha='center', va='center', fontsize=8, color='white',
                               weight='500', family='Segoe UI')
        
        # Plot final segments
        if has_final:
            for seg in final_segments:
                if seg.get('startTimeMs') is not None and seg.get('endTimeMs') is not None:
                    start = seg['startTimeMs'] / 1000
                    duration = (seg['endTimeMs'] - seg['startTimeMs']) / 1000
                    
                    rect = patches.Rectangle(
                        (start, y_positions['final'] - bar_height/2),
                        duration, bar_height,
                        linewidth=1, edgecolor=border_color, facecolor=final_color,
                        alpha=0.9
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    selling_point = seg.get('sellingPoint', '')
                    if selling_point and len(selling_point) > 25:
                        selling_point = selling_point[:22] + '...'
                    if selling_point and duration > 0.5:
                        ax.text(start + duration/2, y_positions['final'], selling_point,
                               ha='center', va='center', fontsize=8, color='white',
                               weight='500', family='Segoe UI')
        
        # Configure plot with Fluent UI styling
        ax.set_xlim(0, max_time_seconds * 1.02)
        ax.set_ylim(*y_lim)
        ax.set_xlabel('Time (seconds)', fontsize=14, color=text_color, 
                     weight='600', family='Segoe UI')
        ax.set_ylabel('Segment Type', fontsize=14, color=text_color, 
                     weight='600', family='Segoe UI')
        ax.set_title('Video Segments Analysis', fontsize=20, fontweight='600', 
                    color=text_color, pad=25)
        
        # Set y-axis labels with Fluent UI typography
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, color=text_color, fontsize=12, family='Segoe UI')
        
        # Style the axes with Fluent UI colors
        ax.spines['bottom'].set_color(border_color)
        ax.spines['left'].set_color(border_color)
        ax.spines['top'].set_color(border_color)
        ax.spines['right'].set_color(border_color)
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.tick_params(colors=text_color, which='both', labelsize=11)
        
        # Add subtle grid
        ax.grid(True, axis='x', alpha=0.3, color=grid_color, linestyle='-', linewidth=0.5)
        ax.grid(True, axis='y', alpha=0.2, color=grid_color, linestyle='-', linewidth=0.5)
        
        # Create legend elements based on what exists
        legend_elements = [
            patches.Patch(facecolor=merged_color, alpha=0.9, label='SellingPoint Segments'),
            patches.Patch(facecolor=overlap_color, alpha=0.8, label='Original Segments')
        ]
        if has_unmerged:
            legend_elements.append(patches.Patch(facecolor=unmerged_color, alpha=0.8, label='Unmerged Segments'))
        if has_final:
            legend_elements.append(patches.Patch(facecolor=final_color, alpha=0.9, label='Final Segments'))
        
        # Position legend to avoid overlap - use bottom right instead of top right
        legend = ax.legend(handles=legend_elements, loc='lower right', 
                         facecolor='white', edgecolor=border_color,
                         labelcolor=text_color, fontsize=11, 
                         framealpha=0.95, shadow=True)
        legend.get_frame().set_linewidth(1)
        
        # Add vertical lines at final segment start/end times
        if has_final:
            # Collect unique timestamps from final segments
            final_timestamps = set()
            for seg in final_segments:
                if seg.get('startTimeMs') is not None:
                    final_timestamps.add(seg['startTimeMs'] / 1000)
                if seg.get('endTimeMs') is not None:
                    final_timestamps.add(seg['endTimeMs'] / 1000)
            
            # Draw vertical lines at each final segment timestamp
            for timestamp in sorted(final_timestamps):
                if timestamp > 0:  # Skip zero
                    ax.axvline(x=timestamp, color='#808080', alpha=0.4, linestyle=':', linewidth=1.2, zorder=1)
        
        # Add statistics box with Fluent UI card styling - position in top left
        total_segments = len(merged_segments) + len(unmerged_segments)
        stats_text = f"Total Segments: {total_segments}\nMerged: {len(merged_segments)} | Unmerged: {len(unmerged_segments)}"
        if has_final:
            stats_text += f" | Final: {len(final_segments)}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=11, color=text_color, verticalalignment='top',
               weight='500', family='Segoe UI',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        alpha=0.95, edgecolor=border_color, linewidth=1))
        
        # Tight layout and save with high quality
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='#f3f2f1', edgecolor='none', 
                   metadata={'Creator': 'Video Segmentation Dashboard'})
        plt.close(fig)
        
        logging.info("Visualization saved to: %s", output_path, extra={"output_file": output_path})
        
    except Exception as e:
        logging.error("Failed to create visualization: %s", str(e), extra={"error": str(e)})
        raise

def generate_thumbnail(video_path: str, thumbnail_path: str, timestamp: float = 3.0) -> bool:
    """
    Generate a thumbnail from a video at a specific timestamp using ffmpeg.
    
    Args:
        video_path: Path to the video file
        thumbnail_path: Path where the thumbnail will be saved
        timestamp: Time in seconds to capture the thumbnail (default: 3.0)
        
    Returns:
        bool: True if thumbnail generation was successful, False otherwise
    """
    try:
        # Ensure thumbnail directory exists
        thumbnail_dir = Path(thumbnail_path).parent
        thumbnail_dir.mkdir(exist_ok=True)
        
        # Generate thumbnail using ffmpeg with proper aspect ratio preservation
        # This will pad with black bars to maintain aspect ratio
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(timestamp),  # Seek to timestamp
            '-vframes', '1',        # Extract 1 frame
            '-q:v', '2',           # High quality
            '-vf', 'scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2:black',
            '-y',                  # Overwrite if exists
            thumbnail_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f"Thumbnail generated successfully: {thumbnail_path}")
            return True
        else:
            logging.error(f"Failed to generate thumbnail: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Error generating thumbnail: {str(e)}")
        return False

def get_video_duration(video_path: str) -> Optional[float]:
    """
    Get video duration in seconds using ffmpeg.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        float: Duration in seconds or None if failed
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
        else:
            logging.error(f"Failed to get video duration: {result.stderr}")
            return None
            
    except Exception as e:
        logging.error(f"Error getting video duration: {str(e)}")
        return None

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Video Analysis Pipeline')
    parser.add_argument('--batch', action='store_true', help='Run in batch mode without web UI')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the web server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the web server to')
    parser.add_argument('--log-level', type=str, default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help='Set the logging level (default: DEBUG)')
    return parser.parse_args()

async def process_all_videos_batch():
    """Process all videos in the inputs directory in batch mode"""
    input_dir = Path("inputs")
    videos = list(input_dir.glob("*.mp4"))
    
    if not videos:
        logging.info("No videos found in inputs directory")
        return
    
    logging.info(f"Found {len(videos)} videos to process")
    
    # First, list available analyzers to help debug baseAnalyzerId issues
    if CONTENT_UNDERSTANDING_API_KEY:
        logging.info("Listing available analyzers...")
        loop = asyncio.get_event_loop()
        available_analyzers = await loop.run_in_executor(
            executor,
            list_available_analyzers,
            CONTENT_UNDERSTANDING_ENDPOINT,
            CONTENT_UNDERSTANDING_API_VERSION,
            CONTENT_UNDERSTANDING_API_KEY
        )
        if available_analyzers:
            analyzer_names = [analyzer.get('name', analyzer.get('id', 'Unknown')) for analyzer in available_analyzers.get('value', [])]
            logging.info("Available base analyzers: %s", analyzer_names)
    
    # Process videos concurrently with a limit
    semaphore = asyncio.Semaphore(3)  # Limit concurrent processing
    
    async def process_with_semaphore(video_path):
        async with semaphore:
            video_name = video_path.name
            logging.info(f"Processing {video_name}")
            await process_video_async(str(video_path), video_name)
    
    # Process all videos
    tasks = [process_with_semaphore(video) for video in videos]
    await asyncio.gather(*tasks)
    
    logging.info("Batch processing completed")

# API Endpoints
@app.get("/api/videos")
async def list_videos() -> list[dict]:          # return plain dicts -> can add extra fields
    """List all mp4 videos in inputs and tell whether results already exist"""
    input_dir = Path("inputs")
    thumbnail_dir = Path("thumbnails")
    thumbnail_dir.mkdir(exist_ok=True)

    videos: list[dict] = []
    for video_path in input_dir.glob("*.mp4"):
        base = video_path.with_suffix("")
        # result json files generated by the pipeline
        results_available = any((base.parent / f"{base.name}{sfx}").exists()
                                for sfx in ("_selling_points.json",
                                            "_merged_segments.json",
                                            "_segments_visualization.png"))

        # thumbnail (generate once if missing)
        thumb_path = thumbnail_dir / f"{base.name}.jpg"
        if not thumb_path.exists():
            generate_thumbnail(str(video_path), str(thumb_path))
        thumbnail_url = f"/api/thumbnail/{video_path.name}" if thumb_path.exists() else None

        videos.append({
            "name": video_path.name,
            "path": str(video_path),
            "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
            "duration": get_video_duration(str(video_path)),
            "thumbnail_url": thumbnail_url,
            "results_available": results_available
        })

    return videos

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file to the inputs directory"""
    # Validate file type
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File type {file_extension} not allowed. Allowed types: {', '.join(allowed_extensions)}")
    
    # Create inputs directory if it doesn't exist
    input_dir = Path("inputs")
    input_dir.mkdir(exist_ok=True)
    
    # Generate unique filename if file already exists
    original_filename = file.filename
    file_path = input_dir / original_filename
    
    # Check if file exists and generate unique name
    if file_path.exists():
        stem = file_path.stem
        suffix = file_path.suffix
        counter = 1
        while file_path.exists():
            file_path = input_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    
    # Save uploaded file
    try:
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Get file info
        file_stat = file_path.stat()
        
        # Get video duration
        duration = get_video_duration(str(file_path))
        
        # Generate thumbnail
        thumbnail_dir = Path("thumbnails")
        thumbnail_dir.mkdir(exist_ok=True)
        thumbnail_path = thumbnail_dir / f"{file_path.stem}.jpg"
        thumbnail_url = None
        
        # Generate thumbnail in background
        success = generate_thumbnail(str(file_path), str(thumbnail_path))
        if success:
            thumbnail_url = f"/api/thumbnail/{file_path.name}"
        
        video_info = VideoInfo(
            name=file_path.name,
            path=str(file_path),
            size_mb=round(file_stat.st_size / (1024 * 1024), 2),
            duration=duration,
            thumbnail_url=thumbnail_url
        )
        
        # Broadcast update to all connected clients
        await manager.broadcast({
            "type": "video_added",
            "video": video_info.dict()
        })
        
        return {
            "message": "Video uploaded successfully",
            "video": video_info,
            "original_filename": original_filename,
            "saved_filename": file_path.name
        }
        
    except Exception as e:
        # If upload fails, try to clean up
        if file_path.exists():
            file_path.unlink()
        logging.error("Failed to upload video: %s", str(e), extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")

@app.post("/api/process")
async def process_video_endpoint(request: ProcessVideoRequest, background_tasks: BackgroundTasks):
    """Start processing a video"""
    video_path = os.path.join("inputs", request.video_name)
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Start processing in background
    background_tasks.add_task(
        process_video_async, 
        video_path, 
        request.video_name
    )
    
    return {"message": "Processing started", "video_name": request.video_name}

@app.post("/api/process-batch")
async def process_batch_endpoint(request: BatchProcessRequest, background_tasks: BackgroundTasks):
    """Start processing multiple videos"""
    processed_videos = []
    not_found_videos = []
    
    for video_name in request.video_names:
        video_path = os.path.join("inputs", video_name)
        
        if not os.path.exists(video_path):
            not_found_videos.append(video_name)
            continue
        
        # Start processing in background
        background_tasks.add_task(
            process_video_async, 
            video_path, 
            video_name
        )
        processed_videos.append(video_name)
    
    return {
        "message": f"Started processing {len(processed_videos)} videos",
        "processed_videos": processed_videos,
        "not_found_videos": not_found_videos
    }

@app.get("/api/status/{video_name}")
async def get_status(video_name: str):
    """Get processing status for a video"""
    if video_name not in processing_status:
        return {"status": "not_started"}
    return processing_status[video_name]

@app.get("/api/status-all")
async def get_all_status():
    """Get processing status for all videos"""
    return processing_status

@app.get("/api/results/{video_name}")
async def get_results(video_name: str):
    """Get processing results for a video"""
    base_path = os.path.join("inputs", os.path.splitext(video_name)[0])
    
    results = {}
    result_files = {
        "word_transcription": f"{base_path}_word.txt",
        "sentence_transcription": f"{base_path}_sentence.txt",
        "selling_points": f"{base_path}_selling_points.json",
        "merged_segments": f"{base_path}_merged_segments.json",
        "content_understanding": f"{base_path}.mp4.json"
    }
    
    for key, file_path in result_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    results[key] = json.load(f)
                else:
                    results[key] = f.read()
    
    # Process content understanding segments for easier display
    if "content_understanding" in results and "result" in results["content_understanding"]:
        # Handle prebuilt-videoAnalyzer format: contents[0].segments[]
        content_result = results["content_understanding"]["result"]
        if "contents" in content_result and content_result["contents"]:
            raw_segments = content_result["contents"][0].get("segments", [])
        else:
            raw_segments = []
        processed_segments = []
        
        for idx, segment in enumerate(raw_segments):
            processed_segment = {
                "index": idx,
                "startTimeMs": segment.get("startTimeMs", 0),
                "endTimeMs": segment.get("endTimeMs", 0),
                "duration": (segment.get("endTimeMs", 0) - segment.get("startTimeMs", 0)) / 1000.0,
                "sellingPoint": "",  # prebuilt-videoAnalyzer doesn't have sellingPoint field
                "description": segment.get("description", ""),
                "confidence": 0,  # prebuilt-videoAnalyzer doesn't provide confidence scores
                "segmentId": segment.get("segmentId", str(idx + 1))
            }
            
            # Add merge status if merged_segments exists
            if "merged_segments" in results:
                # Check if this segment was merged
                is_merged = False
                merged_with = []
                
                for merged_seg in results["merged_segments"].get("merged_segments", []):
                    for overlap_seg in merged_seg.get("overlapping_segments", []):
                        if (overlap_seg["startTimeMs"] == segment.get("startTimeMs") and 
                            overlap_seg["endTimeMs"] == segment.get("endTimeMs")):
                            is_merged = True
                            merged_with.append(merged_seg["content"])
                            break
                
                processed_segment["isMerged"] = is_merged
                processed_segment["mergedWith"] = merged_with
            
            processed_segments.append(processed_segment)
        
        results["content_understanding_segments"] = processed_segments
    
    return results

@app.get("/api/visualization/{video_name}")
async def get_visualization(video_name: str):
    """Get visualization image for a video"""
    base_path = os.path.join("inputs", os.path.splitext(video_name)[0])
    viz_path = f"{base_path}_segments_visualization.png"
    
    if os.path.exists(viz_path):
        return FileResponse(viz_path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Visualization not found")

@app.get("/api/thumbnail/{video_name}")
async def get_thumbnail(video_name: str):
    """Get thumbnail image for a video"""
    # Create thumbnails directory if it doesn't exist
    thumbnail_dir = Path("thumbnails")
    thumbnail_dir.mkdir(exist_ok=True)
    
    # Generate thumbnail filename
    video_base = os.path.splitext(video_name)[0]
    thumbnail_path = thumbnail_dir / f"{video_base}.jpg"
    
    # Check if thumbnail exists
    if not thumbnail_path.exists():
        # Generate thumbnail
        video_path = Path("inputs") / video_name
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Try to generate thumbnail
        success = generate_thumbnail(str(video_path), str(thumbnail_path))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
    
    return FileResponse(str(thumbnail_path), media_type="image/jpeg")

@app.delete("/api/videos/{video_name}")
async def delete_video(video_name: str):
    """Delete a video file and its associated files"""
    video_path = Path("inputs") / video_name
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        # Delete main video file
        video_path.unlink()
        logging.info("Deleted video file: %s", video_name, extra={"video": video_name})
        
        # Delete associated files
        base_path = video_path.with_suffix('')
        associated_files = [
            f"{base_path}_word.txt",
            f"{base_path}_sentence.txt", 
            f"{base_path}_selling_points.json",
            f"{base_path}_merged_segments.json",
            f"{base_path}_segments_visualization.png",
            f"{video_path}.json"  # content understanding results
        ]
        
        deleted_files = []
        for file_path in associated_files:
            if Path(file_path).exists():
                Path(file_path).unlink()
                deleted_files.append(Path(file_path).name)
        
        if deleted_files:
            logging.info("Deleted associated files: %s", ", ".join(deleted_files), extra={"files": deleted_files})
        
        # Delete thumbnail from thumbnails directory
        thumbnail_dir = Path("thumbnails")
        if thumbnail_dir.exists():
            thumbnail_path = thumbnail_dir / f"{base_path.name}.jpg"
            if thumbnail_path.exists():
                thumbnail_path.unlink()
                logging.info("Deleted thumbnail: %s", thumbnail_path.name, extra={"thumbnail": thumbnail_path.name})
            else:
                logging.warning("Thumbnail not found: %s", thumbnail_path.name, extra={"thumbnail": thumbnail_path.name})
        
        # Remove from processing status
        if video_name in processing_status:
            del processing_status[video_name]
            logging.info("Removed from processing status: %s", video_name, extra={"video": video_name})
        
        # Broadcast update to all connected clients
        await manager.broadcast({
            "type": "video_deleted",
            "video_name": video_name
        })
        
        return {"message": f"Video {video_name} and associated files deleted successfully"}
        
    except Exception as e:
        logging.error("Failed to delete video %s: %s", video_name, str(e), extra={"video": video_name, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Serve video files from inputs directory - MUST BE BEFORE static files mount
@app.get("/inputs/{filename}")
async def serve_video(filename: str):
    """Serve video files from the inputs directory"""
    video_path = Path("inputs") / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Determine media type based on file extension
    extension = video_path.suffix.lower()
    media_types = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.webm': 'video/webm'
    }
    
    media_type = media_types.get(extension, 'video/mp4')
    
    return FileResponse(
        path=str(video_path),
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",  # Enable video seeking
            "Content-Disposition": f"inline; filename={filename}"
        }
    )

# Serve static files (frontend) - MUST BE AFTER all other routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    args = parse_arguments()
    
    # Configure logging based on command line argument
    setup_logging(args.log_level)
    
    # Debug proxy and network configuration
    if args.log_level == 'DEBUG':
        logging.debug("HTTP_PROXY: %s", os.environ.get('HTTP_PROXY', 'Not set'))
        logging.debug("HTTPS_PROXY: %s", os.environ.get('HTTPS_PROXY', 'Not set'))
        logging.debug("NO_PROXY: %s", os.environ.get('NO_PROXY', 'Not set'))
        logging.debug("Environment variables loaded:")
        for key, value in [
            ('AZURE_SPEECH_KEY', SPEECH_KEY),
            ('AZURE_SPEECH_ENDPOINT', SPEECH_ENDPOINT),
            ('AZURE_OPENAI_ENDPOINT', OPENAI_ENDPOINT),
            ('AZURE_CONTENT_UNDERSTANDING_ENDPOINT', CONTENT_UNDERSTANDING_ENDPOINT),
        ]:
            logging.debug("%s: %s", key, 'SET' if value else 'NOT SET')
    
    if args.batch:
        # Run in batch mode without web UI
        logging.info("Starting batch processing mode with log level: %s", args.log_level)
        
        # Run the async batch processing
        asyncio.run(process_all_videos_batch())
        
        # Exit after batch processing
        sys.exit(0)
    else:
        # Start the web UI server
        import uvicorn
        logging.info("Starting web UI server on %s:%d with log level: %s", args.host, args.port, args.log_level)
        uvicorn.run(app, host=args.host, port=args.port)