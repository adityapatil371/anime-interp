.PHONY: precompute train evaluate interpolate demo docker-build docker-run

# Precompute distance maps (run once before training)
precompute:
	python scripts/precompute_distances.py --split both --workers 4

# Train the model
train:
	python scripts/train.py --epochs 15 --batch-size 4 --num-workers 4

# Evaluate on test set
evaluate:
	python scripts/evaluate.py --unet-checkpoint checkpoints/unet_best.pth

# Interpolate two frames
# Usage: make interpolate FRAME_A=a.jpg FRAME_B=b.jpg OUTPUT=out.png
interpolate:
	python scripts/interpolate.py \
		--frame-a $(FRAME_A) \
		--frame-b $(FRAME_B) \
		--output $(OUTPUT)

# Launch Gradio demo
demo:
	python app.py --unet-checkpoint checkpoints/unet_best.pth

# Docker
docker-build:
	docker build -t anime-interp .

docker-run:
	docker run -p 7860:7860 anime-interp