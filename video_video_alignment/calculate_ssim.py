from ffmpeg_quality_metrics import FfmpegQualityMetrics



def calculate_ssim(generator_video, target_video):
    ffqm = FfmpegQualityMetrics(generator_video, target_video)

    metrics = ffqm.calculate(["ssim"])


    return sum([frame["ssim_y"] for frame in metrics["ssim"]]) / len(metrics["ssim"])