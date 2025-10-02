"""
Main Entry Point cho Pickleball Video Analysis Pipeline

Script chính để chạy toàn bộ pipeline từ video input đến visualization output.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import (
    load_config, setup_logging, create_output_dirs, validate_video_files,
    Timer, save_tracking_data
)
from src.court_detection import CourtDetector
from src.detection import PickleballDetector
from src.tracking import PickleballTracker
from src.analysis import MovementAnalyzer
from src.visualization import PickleballVisualizer

from loguru import logger

class PickleballPipeline:
    """
    Main pipeline class để xử lý toàn bộ workflow.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Khởi tạo pipeline.
        
        Args:
            config_path: Đường dẫn đến file cấu hình
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        setup_logging(self.config)
        
        # Create output directories
        self.output_dirs = create_output_dirs(
            self.config['output']['base_dir'], 
            self.config
        )
        
        # Initialize components
        self.court_detector = CourtDetector(config_path)
        self.detector = PickleballDetector(self.config)
        self.tracker = PickleballTracker(self.config)
        self.analyzer = MovementAnalyzer(self.config)
        self.visualizer = PickleballVisualizer(self.config)
        
        logger.info("Pickleball Pipeline đã được khởi tạo")

    def run_full_pipeline(self, 
                         video_paths: Optional[List[str]] = None,
                         max_frames: Optional[int] = None,
                         skip_calibration: bool = False) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline từ đầu đến cuối.
        
        Args:
            video_paths: Danh sách đường dẫn video (None = sử dụng từ config)
            max_frames: Số frame tối đa để xử lý (None = tất cả)
            skip_calibration: Bỏ qua bước calibration court
            
        Returns:
            Dictionary chứa kết quả pipeline
        """
        logger.info("🚀 BẮT ĐẦU CHẠY PICKLEBALL ANALYSIS PIPELINE")
        start_time = time.time()
        
        results = {
            'status': 'started',
            'pipeline_stages': {},
            'output_files': {},
            'statistics': {},
            'errors': []
        }
        
        try:
            # Stage 1: Validate and prepare videos
            with Timer("Stage 1: Video validation"):
                if video_paths is None:
                    video_paths = [
                        os.path.join("..", path) for path in self.config['video']['input_videos']
                    ]
                
                valid_videos = validate_video_files(video_paths)
                if not valid_videos:
                    raise ValueError("Không có video hợp lệ để xử lý")
                
                results['pipeline_stages']['video_validation'] = {
                    'status': 'completed',
                    'input_videos': len(video_paths),
                    'valid_videos': len(valid_videos),
                    'video_files': valid_videos
                }
                
                logger.info(f"✅ Stage 1 hoàn thành: {len(valid_videos)} video hợp lệ")
            
            # Stage 2: Court calibration (if needed)
            if not skip_calibration:
                with Timer("Stage 2: Court calibration check"):
                    calibration_status = self.court_detector.validate_calibration()
                    uncalibrated = [cam for cam, status in calibration_status.items() if not status]
                    
                    if uncalibrated:
                        logger.warning(f"Camera chưa calibrate: {uncalibrated}")
                        logger.info("Chạy: 'python src/court_detection.py --calibrate' để calibrate")
                        
                        # Auto calibrate nếu chỉ có 1 video
                        if len(valid_videos) == 1:
                            camera_name = 'san1'  # Default cho video đầu tiên
                            logger.info(f"Auto calibrating {camera_name}...")
                            self.court_detector.calibrate_camera(valid_videos[0], camera_name)
                    
                    results['pipeline_stages']['court_calibration'] = {
                        'status': 'completed',
                        'calibration_status': calibration_status,
                        'uncalibrated_cameras': uncalibrated
                    }
                    
                    logger.info(f"✅ Stage 2 hoàn thành: Calibration checked")
            
            # Stage 3: Object detection
            with Timer("Stage 3: Object detection"):
                all_detections = []
                
                for i, video_path in enumerate(valid_videos[:1]):  # Process first video for demo
                    logger.info(f"Detecting objects trong {os.path.basename(video_path)}")
                    
                    video_detections = self.detector.detect_video(
                        video_path, 
                        max_frames=max_frames
                    )
                    
                    all_detections.extend(video_detections)
                
                # Save detection data
                detection_stats = self.detector.get_detection_statistics(all_detections)
                
                results['pipeline_stages']['object_detection'] = {
                    'status': 'completed',
                    'total_frames': len(all_detections),
                    'statistics': detection_stats
                }
                
                logger.info(f"✅ Stage 3 hoàn thành: {len(all_detections)} frames detected")
            
            # Stage 4: Multi-object tracking
            with Timer("Stage 4: Multi-object tracking"):
                tracking_results = self.tracker.process_video_detections(all_detections)
                
                # Export tracking data
                tracking_csv_path = os.path.join(
                    self.output_dirs['data'], 'tracking_results.csv'
                )
                tracking_df = self.tracker.export_tracking_data(tracking_csv_path)
                
                tracking_stats = self.tracker.get_tracking_statistics()
                
                results['pipeline_stages']['tracking'] = {
                    'status': 'completed',
                    'tracking_records': len(tracking_df),
                    'statistics': tracking_stats
                }
                
                results['output_files']['tracking_data'] = tracking_csv_path
                
                logger.info(f"✅ Stage 4 hoàn thành: {len(tracking_df)} tracking records")
            
            # Stage 5: Movement analysis
            with Timer("Stage 5: Movement analysis"):
                analysis_results = self.analyzer.analyze_tracking_data(tracking_df)
                
                # Export analysis results
                analysis_json_path = os.path.join(
                    self.output_dirs['reports'], 'analysis_results.json'
                )
                self.analyzer.export_analysis_results(analysis_results, analysis_json_path)
                
                results['pipeline_stages']['analysis'] = {
                    'status': 'completed',
                    'players_analyzed': len(analysis_results.get('player_analysis', {})),
                    'balls_analyzed': len(analysis_results.get('ball_analysis', {}))
                }
                
                results['output_files']['analysis_data'] = analysis_json_path
                
                logger.info(f"✅ Stage 5 hoàn thành: Analysis completed")
            
            # Stage 6: Visualization
            with Timer("Stage 6: Visualization"):
                viz_output_dir = self.output_dirs['charts']
                viz_files = self.visualizer.save_all_visualizations(
                    analysis_results, viz_output_dir
                )
                
                results['pipeline_stages']['visualization'] = {
                    'status': 'completed',
                    'visualizations_created': len(viz_files)
                }
                
                results['output_files'].update(viz_files)
                
                logger.info(f"✅ Stage 6 hoàn thành: {len(viz_files)} visualizations created")
            
            # Final summary
            total_time = time.time() - start_time
            results['status'] = 'completed'
            results['statistics'] = {
                'total_execution_time': total_time,
                'videos_processed': len(valid_videos),
                'frames_analyzed': len(all_detections),
                'tracking_records': len(tracking_df),
                'output_files_created': len(results['output_files'])
            }
            
            logger.info(f"🎉 PIPELINE HOÀN THÀNH trong {total_time:.2f} giây")
            self._print_summary(results)
            
        except Exception as e:
            logger.error(f"❌ Pipeline thất bại: {e}")
            results['status'] = 'failed'
            results['errors'].append(str(e))
            raise
        
        return results

    def run_detection_only(self, 
                          video_path: str, 
                          max_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Chỉ chạy detection cho một video.
        
        Args:
            video_path: Đường dẫn video
            max_frames: Số frame tối đa
            
        Returns:
            List detection results
        """
        logger.info(f"Chạy detection only cho {video_path}")
        
        detections = self.detector.detect_video(video_path, max_frames=max_frames)
        
        # Save detection video
        output_video_path = os.path.join(
            self.output_dirs['videos'], 
            f"detection_{os.path.basename(video_path)}"
        )
        self.detector.save_detection_video(video_path, detections, output_video_path)
        
        logger.info(f"Detection completed: {len(detections)} frames")
        return detections

    def run_tracking_only(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Chỉ chạy tracking với detection data có sẵn.
        
        Args:
            detections: Detection data
            
        Returns:
            Tracking results
        """
        logger.info("Chạy tracking only")
        
        tracking_results = self.tracker.process_video_detections(detections)
        
        # Export data
        tracking_csv_path = os.path.join(
            self.output_dirs['data'], 'tracking_only_results.csv'
        )
        tracking_df = self.tracker.export_tracking_data(tracking_csv_path)
        
        return {
            'tracking_results': tracking_results,
            'tracking_data': tracking_df,
            'output_file': tracking_csv_path
        }

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """
        In summary kết quả pipeline.
        
        Args:
            results: Kết quả pipeline
        """
        print("\n" + "="*60)
        print("🏓 PICKLEBALL ANALYSIS PIPELINE SUMMARY")
        print("="*60)
        
        stats = results.get('statistics', {})
        print(f"⏱️  Execution Time: {stats.get('total_execution_time', 0):.2f} seconds")
        print(f"🎥 Videos Processed: {stats.get('videos_processed', 0)}")
        print(f"🖼️  Frames Analyzed: {stats.get('frames_analyzed', 0)}")
        print(f"📊 Tracking Records: {stats.get('tracking_records', 0)}")
        print(f"📁 Output Files: {stats.get('output_files_created', 0)}")
        
        print("\n📂 Output Files Created:")
        for file_type, file_path in results.get('output_files', {}).items():
            print(f"  • {file_type}: {file_path}")
        
        print("\n🎯 Pipeline Stages:")
        for stage, stage_data in results.get('pipeline_stages', {}).items():
            status = stage_data.get('status', 'unknown')
            emoji = "✅" if status == 'completed' else "❌"
            print(f"  {emoji} {stage}: {status}")
        
        print("\n" + "="*60)

def main():
    """Main function với command line interface."""
    parser = argparse.ArgumentParser(
        description="Pickleball Video Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chạy full pipeline
  python main.py --input data_video/ --output output/
  
  # Chạy với giới hạn frames
  python main.py --input data_video/ --max-frames 500
  
  # Chỉ chạy detection
  python main.py --mode detection --input data_video/san1.mp4
  
  # Chỉ chạy tracking với detection data có sẵn
  python main.py --mode tracking --input output/tracking_data/detections.json
        """
    )
    
    parser.add_argument('--input', '-i', 
                       help='Đường dẫn video hoặc thư mục chứa video')
    parser.add_argument('--output', '-o', default='output',
                       help='Thư mục output (default: output)')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='File cấu hình (default: config/config.yaml)')
    parser.add_argument('--mode', choices=['full', 'detection', 'tracking', 'analysis'],
                       default='full', help='Chế độ chạy pipeline')
    parser.add_argument('--max-frames', type=int,
                       help='Số frame tối đa để xử lý')
    parser.add_argument('--skip-calibration', action='store_true',
                       help='Bỏ qua bước kiểm tra calibration')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Hiển thị log chi tiết')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = PickleballPipeline(args.config)
        
        if args.verbose:
            logger.remove()
            logger.add(lambda msg: print(msg, end=""), level="DEBUG")
        
        # Determine input videos
        video_paths = None
        if args.input:
            if os.path.isfile(args.input):
                video_paths = [args.input]
            elif os.path.isdir(args.input):
                video_paths = []
                for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                    video_paths.extend(Path(args.input).glob(ext))
                video_paths = [str(p) for p in video_paths]
        
        # Run pipeline based on mode
        if args.mode == 'full':
            results = pipeline.run_full_pipeline(
                video_paths=video_paths,
                max_frames=args.max_frames,
                skip_calibration=args.skip_calibration
            )
        elif args.mode == 'detection':
            if not video_paths or len(video_paths) == 0:
                raise ValueError("Cần specify video path cho detection mode")
            results = pipeline.run_detection_only(video_paths[0], args.max_frames)
        else:
            raise ValueError(f"Mode '{args.mode}' chưa được implement đầy đủ")
        
        logger.info("Pipeline completed successfully! 🎉")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline bị ngắt bởi user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()