from modules.watermark.image_watermark.raw import RAWProcessor

def image_watermark_pipeline(args):
    algorithm = args.algorithm.lower()
    task = args.task.lower()

    if algorithm == "raw":
        processor = RAWProcessor(args)

        if task == 'add_watermark':
            if args.media_type == 'image':
                processor.add_watermark_image(args.image_path, out_prefix='image')
            elif args.media_type == 'video':
                processor.add_watermark_video(args.video_dir, num_frames=args.num_frames, crop_size=args.crop_size, output_dir=args.output_dir)
            else:
                raise ValueError(f"Unsupported media_type: {args.media_type}")

        elif task == 'detect':
            if args.media_type == 'image':
                processor.detect_image(args.image_path)
            elif args.media_type == 'video':
                processor.detect_video(args.video_dir, num_frames=args.num_frames, crop_size=args.crop_size)
            else:
                raise ValueError(f"Unsupported media_type: {args.media_type}")

        elif task == 'evaluate':
            if args.media_type == 'image':
                processor.evaluate_image(args.image_path, args.wt_image_path, args.metric)
            elif args.media_type == 'video':
                processor.evaluate_video(args.video_dir, args.video_dir_wt, num_frames=args.num_frames, crop_size=args.crop_size, metric=args.metric)

        else:
            raise ValueError(f"Unsupported task type: {args.task}")
    else:
        raise NotImplementedError(f"Algorithm '{args.algorithm}' not supported.")
