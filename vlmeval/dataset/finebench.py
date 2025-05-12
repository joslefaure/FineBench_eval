import os
import re
import json
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.finebench import get_dimension_rating, extract_characters_regex, extract_option

FAIL_MSG = 'Failed to obtain answer via API.'


class FineBench(VideoBaseDataset):
    """Dataset for handling video-based action question answering tasks with streaming frames."""

    BENCHMARK_TYPE = "mcqa"
    MD5 = None  # No MD5 check for streaming data
    TYPE = 'Video-MCQ'

    def __init__(self, dataset='FineBench', data_root="./", buffer_size=8, use_subtitle=False, 
                fps=-1, nframe=8, max_frames=32, spatial_grounding=True, stride=1, center_frame_only=False, mode='streaming'):
        """
        Initialize the streaming video dataset.

        Args:
            dataset: Name of the dataset
            buffer_size: Size of the frame buffer
            use_subtitle: Whether to include subtitles
            fps: Frames per second to sample
            max_frames: Maximum number of frames to use
            stride: Step size for sliding windows
            center_frame_only: If True, only questions for center frames are used
            mode: 'streaming' or 'window' - streaming shows historical frames with target as last frame,
                window shows frames centered around target
        """
        self.buffer_size = buffer_size
        self.use_subtitle = use_subtitle
        self.dataset_name = dataset
        self.stride = stride
        self.center_frame_only = center_frame_only
        self.img_root = "./tmp_img"
        self.mode = mode
        self.spatial_grounding = spatial_grounding
        self.data_root = data_root
        assert mode in ['streaming', 'window'], "Mode must be either 'streaming' or 'window'"

        FRAMES_TMPL_NOSUB = """
These are the frames of a video stream. \
Select the best answer to the following multiple-choice question based on the most recent frame (frame {current_frame} of {total_frames}). \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

        FRAMES_TMPL_NOSUB_MID = """
These are the frames of a video stream. \
Select the best answer to the following multiple-choice question based on the middle frame (frame {current_frame} of {total_frames}). \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

        FRAMES_TMPL_SUB = """
These are the frames of a video stream. \
This video's subtitles are listed below:
{}
Select the best answer to the following multiple-choice question based on the most recent frame (frame {current_frame} of {total_frames}). \
Respond with only the letter (A, B, C, or D) of the correct option.
"""
        if mode == 'streaming':
            self.FRAMES_TMPL = FRAMES_TMPL_NOSUB
        elif mode == 'window':
            self.FRAMES_TMPL = FRAMES_TMPL_NOSUB_MID
        super().__init__(dataset=dataset, nframe=max_frames, fps=fps)
            
    @classmethod
    def supported_datasets(cls):
        return ['FineBench']

    def prepare_dataset(self, dataset_name='FineBench', repo_id=None):
        """
        Prepare dataset from local files instead of downloading.
        
        Args:
            dataset_name: Name of the dataset
            repo_id: Not used, as we're loading data locally
            
        Returns:
            Dictionary containing data file path and root directory
        """
        # Define data path
        data_file = os.path.join(self.data_root, "annotations/combined_first_7_template_v2.1.json")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        print(f"Loading data from {data_file}")
        return dict(data_file=data_file, root=self.data_root)

    def load_data(self, data_file):
        """Load dataset from JSON file."""
        with open(data_file, 'r') as f:
            data = json.load(f)
            
        frames_folder = os.path.join(self.data_root, "frames_val_v1")
        
        # Group data by video_id
        video_data = defaultdict(list)
        for item in data:
            video_id = item["video_id"]
            video_data[video_id].append(item)
            
        # For each video, sort by timestamp
        for video_id in video_data:
            video_data[video_id].sort(key=lambda x: int(x["timestamp"]))
            
        processed_data = []
        idx = 0
        
        for video_id, items in tqdm(video_data.items(), desc="Processing videos"):
            # Find all available frames for this video by checking the directory
            video_frames_dir = os.path.join(frames_folder, video_id)
            if not os.path.exists(video_frames_dir):
                print(f"Warning: Video directory not found: {video_frames_dir}")
                continue
                
            # Get all frame files and extract their timestamps
            all_timestamps = []
            for frame_file in os.listdir(video_frames_dir):
                if frame_file.endswith('.jpg'):
                    try:
                        timestamp = int(frame_file.split('.')[0])
                        all_timestamps.append(timestamp)
                    except ValueError:
                        continue
            
            all_timestamps.sort()
            
            # Group items with annotations by timestamp
            timestamp_to_items = defaultdict(list)
            for item in items:
                timestamp_to_items[int(item["timestamp"])].append(item)
                
            # For each timestamp with a question
            for ts in sorted(timestamp_to_items.keys()):
                # Find the index of this timestamp in all_timestamps
                try:
                    target_idx = all_timestamps.index(ts)
                except ValueError:
                    print(f"Warning: Timestamp {ts} not found in video {video_id}")
                    continue
                
                # Calculate buffer indices based on mode
                if self.mode == 'streaming':
                    # Streaming mode: look back to get buffer_size frames (including current)
                    start_idx = max(0, target_idx - (self.buffer_size - 1))
                    end_idx = target_idx + 1  # exclusive
                else:  # window mode
                    # Window mode: center the target frame with buffer_size/2 on each side
                    half_buffer = self.buffer_size // 2
                    start_idx = max(0, target_idx - half_buffer)
                    end_idx = min(len(all_timestamps), target_idx + half_buffer + 1)  # +1 for exclusive
                
                # Get buffer timestamps from all available frames
                buffer_timestamps = all_timestamps[start_idx:end_idx]
                
                # Collect frame paths for this window
                frame_paths = []
                for buffer_ts in buffer_timestamps:
                    path = os.path.join(frames_folder, f"{video_id}/{buffer_ts}.jpg")
                    if not os.path.exists(path):
                        print(f"Warning: Cannot find frame {path}")
                        continue  # Continue loading other frames even if one is missing
                    frame_paths.append(path)
                
                # Skip windows with no frames at all
                if len(frame_paths) == 0:
                    continue
                
                # Set target frame index based on mode
                if self.mode == 'streaming':
                    # Target frame is the most recent/last one in streaming mode
                    relative_target_idx = len(frame_paths) - 1
                else:  # window mode
                    # Target frame should be in the middle
                    # Calculate position of target frame in the actual buffer
                    relative_target_idx = frame_paths.index(os.path.join(frames_folder, f"{video_id}/{ts}.jpg"))
                
                # Process all questions at this timestamp
                for item in timestamp_to_items[ts]:
                    row = {
                        "index": idx,
                        "video": video_id,
                        "video_path": None,  # Not using video path
                        "frame_paths": frame_paths,
                        "buffer_timestamps": buffer_timestamps,
                        "target_frame_idx": relative_target_idx,
                        "timestamp": str(ts),
                        "question_id": item.get("question_id", str(idx)),
                        "question": item["question"],
                        "candidates": item["options"],
                        "answer": item["answer"],
                        "action_id": item.get("action_id", ""),
                        "action_type": item.get("action_type", ""),
                        "bbox": item.get("bbox", None),
                        "subtitle_path": None
                    }
                    processed_data.append(row)
                    idx += 1
                    
        df = pd.DataFrame(processed_data)
        print(f"Total examples: {len(df)}")
        return df
    
    def dump_image(self, line):
        # print(line)
        os.makedirs(self.img_root, exist_ok=True)

        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(self.img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        return tgt_path

    def save_video_frames(self, frame_paths, video_llm=False):
        """Return the frame paths directly without saving, as they're already on disk."""
        return frame_paths, None, {"fps": self.fps, "n_frames": len(frame_paths)}

    def build_prompt(self, line, video_llm=False):
        # video_llm = False  # Not using video LLM for streaming data
        """Build prompt for video streaming task."""
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
            
        frame_paths = line['frame_paths']
        buffer_size = len(frame_paths)
        target_idx = line['target_frame_idx']
        
        # For subtitles
        subtitles = ""
        if self.use_subtitle and line['subtitle_path'] and os.path.exists(line['subtitle_path']):
            import pysubs2
            subs = pysubs2.load(line['subtitle_path'], encoding='utf-8')
            subtitle_texts = []
            
            for ts in line['buffer_timestamps']:
                sub_text = ''
                cur_time = pysubs2.make_time(seconds=int(ts)/self.fps)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace('\\N', ' ')
                        break
                if sub_text.strip():
                    subtitle_texts.append(sub_text)
            subtitles = '\n'.join(subtitle_texts)
            
        message = [dict(type='text', value="")]  # Empty system message
        
        if video_llm:
            # For video LLM, we'd need to create a video from frames
            # This is a placeholder - would need to stitch frames into a video
            # message.append(dict(type='video', value=video_path))
            warnings.warn("Warning: video_llm=True not fully implemented for streaming dataset.")
            for im in frame_paths:
                message.append(dict(type='image', value=im))
        else:
            for im in frame_paths:
                message.append(dict(type='image', value=im))
                
        text_prompt = self.FRAMES_TMPL.format(current_frame=target_idx+1, total_frames=buffer_size) if not self.use_subtitle else self.FRAMES_TMPL.format(subtitles, current_frame=target_idx+1, total_frames=buffer_size)
        
        message.append(dict(type='text', value=text_prompt))
        
        # Add bounding box info if available
        if self.spatial_grounding:
            bbox_prompt = ""
            if line['bbox'] is not None:
                x_min, y_min, x_max, y_max = line['bbox']
                bbox_prompt = f"Pay attention to the region within the bounding box coordinates [x_min={x_min:.3f}, y_min={y_min:.3f}, x_max={x_max:.3f}, y_max={y_max:.3f}] in the target frame.\n"
                message.append(dict(type='text', value=bbox_prompt))
            
        # Format question with options
        question = line['question']
        options = line['candidates']
        if isinstance(options, str):
            options = eval(options)
        prompt = f'Question: {question}\n' + '\n'.join(options) + '\nAnswer: '
        message.append(dict(type='text', value=prompt))
        
        # print(message)
        # print("Num of frames: ", len(frame_paths))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """Evaluate model predictions."""
        
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
        
        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')
        
        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')
            assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
            
            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
                
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}
            
            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]
            
            for idx in data.index:
                ans = data.loc[idx, 'answer']
                pred = str(data.loc[idx, 'prediction'])
                
                if extract_characters_regex(pred) == '':
                    extract_pred = extract_option(
                        model,
                        data.loc[idx].to_dict(),
                        'FineBench'
                    )
                    data.loc[idx, 'score'] = int(extract_pred == ans)
                else:
                    data.loc[idx, 'score'] = int(extract_characters_regex(pred) == ans)
                    
            rejected = [x for x in data['score'] if x == -1]
            
            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )
            
            dump(data, score_file)
            
        # Compute ratings by action type
        data = load(score_file)
        
        # Overall rating
        overall_rating = get_dimension_rating(score_file)
        
        # Group by action type
        ratings = {"overall": overall_rating}
        
        if 'action_type' in data.columns:
            action_types = data['action_type'].unique()
            for action_type in action_types:
                action_data = data[data['action_type'] == action_type].copy()
                if len(action_data) > 0:
                    action_score_file = score_file.replace('_score.xlsx', f'_{action_type}_score.xlsx')
                    dump(action_data, action_score_file)
                    action_rating = get_dimension_rating(action_score_file)
                    ratings[action_type] = action_rating
        
        dump(ratings, tgt_file)
        return ratings