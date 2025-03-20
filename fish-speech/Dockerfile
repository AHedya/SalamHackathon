FROM fishaudio/fish-speech:latest-dev 

CMD ["python", "-m", "tools.api_server", \
    "--listen", "0.0.0.0:8080", \
    "--llama-checkpoint-path", "checkpoints/fish-speech-1.5", \
    "--decoder-checkpoint-path", "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth", \
    "--decoder-config-name", "firefly_gan_vq"]

#docker run -it --name slam-hack-tts-container --gpus all -p 8080:8080 slam-hack-tts zsh
# python -m tools.api_server \
#     --listen 0.0.0.0:7860 \
#     --llama-checkpoint-path "checkpoints/fish-speech-1.5" \
#     --decoder-checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
#     --decoder-config-name firefly_gan_vq