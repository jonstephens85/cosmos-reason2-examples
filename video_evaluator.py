import subprocess
import json
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import re

def evaluate_video(video_path):
    """Evaluate a single video - get clean output"""
    cmd = [
        "cosmos-reason2-inference", "online",  # Removed -v flag
        "--port", "8000",
        "--prompt", "Does this video conform to real world physics? Analyze the motion, interactions, and physical behaviors shown. Point out any violations of physical laws or unrealistic elements.",
        "--reasoning",
        "--videos", str(video_path),
        "--fps", "4"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    # Parse the cleaner output - look for Assistant: section
    assistant_match = re.search(r'Assistant:\s*(\w+)', output)
    reasoning_match = re.search(r'Reasoning:\s*(.*?)(?:--------------------\s*Assistant:|$)', output, re.DOTALL)
    
    if assistant_match:
        answer = assistant_match.group(1).strip()
        answer_upper = answer.upper()
    else:
        answer = "UNKNOWN"
        answer_upper = "UNKNOWN"
    
    # Get reasoning for context
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = "N/A"
    
    # Decision based on answer
    if 'YES' in answer_upper:
        verdict = "PASS"
    elif 'NO' in answer_upper:
        verdict = "FAIL"
    else:
        verdict = "UNCLEAR"
    
    return verdict, output, answer, reasoning

def batch_evaluate(video_folder, output_folder=None, verbose=False):
    """Batch evaluate all videos in folder"""
    video_folder = Path(video_folder)
    
    # Use input folder for output if not specified
    if output_folder is None:
        output_folder = video_folder
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    videos = list(video_folder.glob("*.mp4"))
    
    if not videos:
        print(f"?  No MP4 files found in {video_folder}")
        return None
    
    results = []
    
    print(f"? Processing {len(videos)} videos from: {video_folder}")
    print(f"? Output will be saved to: {output_folder}\n")
    
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] Evaluating: {video.name}")
        verdict, full_output, answer, reasoning = evaluate_video(video)
        
        results.append({
            'video': video.name,
            'video_path': str(video),
            'verdict': verdict,
            'answer': answer,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"  ? Answer: {answer}")
        print(f"  ? Verdict: {verdict}")
        
        # Show snippet if verbose
        if verbose:
            print(f"  ? Reasoning: {reasoning[:200]}...")
        print()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV in the specified output folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_folder / f"evaluation_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    total = len(df)
    passed = (df['verdict'] == 'PASS').sum()
    failed = (df['verdict'] == 'FAIL').sum()
    unclear = (df['verdict'] == 'UNCLEAR').sum()
    
    print(f"Total:      {total}")
    print(f"? PASS:    {passed} ({passed/total*100:.1f}%)")
    print(f"? FAIL:    {failed} ({failed/total*100:.1f}%)")
    print(f"? UNCLEAR: {unclear} ({unclear/total*100:.1f}%)")
    print("="*50)
    
    # Show which videos passed vs failed
    print("\n? PASSED:")
    for _, row in df[df['verdict'] == 'PASS'].iterrows():
        print(f"   ? {row['video']}")
    
    print("\n? FAILED:")
    for _, row in df[df['verdict'] == 'FAIL'].iterrows():
        print(f"   ? {row['video']}")
    
    if unclear > 0:
        print("\n? UNCLEAR:")
        for _, row in df[df['verdict'] == 'UNCLEAR'].iterrows():
            print(f"   ? {row['video']}")
    
    print(f"\n? Results saved to: {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluate videos for physics compliance using Cosmos Reason 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_evaluator.py -i assets/batch
  python video_evaluator.py -i test_videos -v
  python video_evaluator.py -i /path/to/videos -o /path/to/results
        """
    )
    
    parser.add_argument(
        '-i', '--input-folder',
        type=str,
        required=True,
        help='Path to folder containing MP4 videos to evaluate'
    )
    
    parser.add_argument(
        '-o', '--output-folder',
        type=str,
        default=None,
        help='Path to folder for output CSV (default: same as input folder)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show reasoning snippets during processing'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='vLLM server port (default: 8000)'
    )
    
    args = parser.parse_args()
    
    # Run batch evaluation
    df = batch_evaluate(args.input_folder, args.output_folder, args.verbose)
    
    if df is not None:
        print("\n? Evaluation complete!")

if __name__ == "__main__":
    main()