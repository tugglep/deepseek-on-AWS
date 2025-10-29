# DeepSeek-OCR on Amazon SageMaker

Deploy the **DeepSeek-OCR** vision-language model on Amazon SageMaker real-time endpoints using a Bring-Your-Own-Container (BYOC) approach with PyTorch and Transformers.

## Features

- **PyTorch/Transformers backend**: Official DeepSeek-OCR implementation
- **Multi-format support**: Single images, multi-page PDFs, URLs, base64, S3 URIs
- **Production-ready**: SageMaker hosting contract (`/ping`, `/invocations`)
- **Flexible prompts**: Free OCR or markdown conversion with bounding boxes
- **Easy deployment**: CodeBuild automation or local Docker build

## What is DeepSeek-OCR?

DeepSeek-OCR is a state-of-the-art vision-language model for optical character recognition tasks. It excels at:
- Extracting text from documents, invoices, receipts, and forms
- Processing handwritten content from whiteboards and notes
- Converting documents to structured Markdown format
- Providing bounding box coordinates for detected text (grounding mode)

**Model Card**: [deepseek-ai/DeepSeek-OCR on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP/JSON
       ▼
┌─────────────────────┐
│  SageMaker Endpoint │
│   (ml.g5.2xlarge)   │
│  ┌───────────────┐  │
│  │  FastAPI      │  │
│  │  (port 8080)  │  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │ Transformers  │  │
│  │ DeepSeek-OCR  │  │
│  │   Model       │  │
│  └───────────────┘  │
└─────────────────────┘
```

## Quick Start

### Prerequisites

- AWS Account with SageMaker permissions
- AWS CLI configured
- Docker (for local builds) or CodeBuild (for AWS builds)

### Option A: CodeBuild (Recommended)

Build and push the container image using AWS CodeBuild:

```bash
cd "DeepSeek OCR/scripts"
./run_codebuild_simple.sh
```

This will:
1. Upload source code to S3
2. Trigger CodeBuild project
3. Build Docker image (~15-20 minutes)
4. Push to Amazon ECR

### Option B: Local Docker Build

Build and push the container locally:

```bash
# Set variables
export AWS_REGION=us-west-2
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPO=deepseek-ocr-sagemaker-byoc

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Create repository if needed
aws ecr create-repository --repository-name $REPO --region $AWS_REGION 2>/dev/null || true

# Build and push
cd "DeepSeek OCR"
docker build -t $REPO:latest -f container/Dockerfile .
docker tag $REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
```

### Deploy to SageMaker

Open and run the demo notebook:

```
notebooks/deepseek_ocr_demo.ipynb
```

The notebook walks through:
1. Creating a SageMaker model
2. Configuring the endpoint
3. Deploying the endpoint (~5-10 minutes)
4. Testing with images and PDFs
5. Cleaning up resources

## Usage Examples

### Single Image OCR

```python
import boto3
import json
import base64

runtime = boto3.client('sagemaker-runtime')

# Read image
with open("invoice.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode("utf-8")

# Invoke endpoint
payload = {
    "prompt": "<image>\\nFree OCR.",
    "image_base64": img_base64
}

response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps(payload)
)

result = json.loads(response['Body'].read())
print(result['text'])
```

### Image from URL or S3

```python
# From HTTP URL
payload = {
    "prompt": "<image>\\nFree OCR.",
    "image_url": "https://example.com/document.jpg"
}

# From S3 URI
payload = {
    "prompt": "<image>\\nFree OCR.",
    "image_url": "s3://my-bucket/documents/invoice.jpg"
}
```

### PDF Processing (Small PDFs)

```python
# Process a 1-2 page PDF
payload = {
    "prompt": "<image>\\nFree OCR.",
    "pdf_url": "https://example.com/document.pdf"
}

response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps(payload)
)

result = json.loads(response['Body'].read())
print(f"Processed {result['pages']} pages")
print(result['text'])
```

**Note**: Real-time endpoints have a 60-second timeout. For PDFs with 3+ pages, process pages individually (see notebook for example).

### Markdown with Bounding Boxes (Grounding Mode)

```python
payload = {
    "prompt": "<image>\\n<|grounding|>Convert the document to markdown.",
    "image_base64": img_base64
}
```

This returns structured markdown with bounding box coordinates in format:
```
<|ref|>text<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
Header Text
```

## API Reference

### Endpoint: `/ping`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "deepseek-ai/DeepSeek-OCR",
  "backend": "transformers"
}
```

### Endpoint: `/invocations`

**Request Format:**
```json
{
  "prompt": "<image>\\nFree OCR.",
  "image_url": "https://example.com/image.jpg",    // OR
  "image_base64": "base64-encoded-image-data",     // OR
  "pdf_url": "https://example.com/doc.pdf",        // OR
  "pdf_base64": "base64-encoded-pdf-data"
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Optional | OCR instruction. Defaults to markdown conversion. Must start with `<image>` token. |
| `image_url` | string | Conditional | URL to image (http://, https://, or s3://) |
| `image_base64` | string | Conditional | Base64-encoded image data |
| `pdf_url` | string | Conditional | URL to PDF file |
| `pdf_base64` | string | Conditional | Base64-encoded PDF data |

**Prompt Options:**
- `"<image>\\nFree OCR."` - Plain text extraction
- `"<image>\\n<|grounding|>Convert the document to markdown."` - Markdown with bounding boxes

**Response Format:**
```json
{
  "text": "Extracted text content...",
  "pages": 3  // Only present for PDF inputs
}
```

## Environment Variables

Configure the container at deployment time:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `deepseek-ai/DeepSeek-OCR` | HuggingFace model ID |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Enable fast model downloads with `hf-transfer` |
| `TRANSFORMERS_CACHE` | `/opt/hf` | Model cache directory |

## Instance Recommendations

| Use Case | Instance Type | GPU | vCPU | Memory | Cost/Hour* |
|----------|---------------|-----|------|--------|------------|
| Development/Testing | `ml.g5.xlarge` | 1x A10G (24GB) | 4 | 16GB | ~$1.41 |
| Production | `ml.g5.2xlarge` | 1x A10G (24GB) | 8 | 32GB | ~$1.52 |
| High Traffic | `ml.g5.4xlarge` | 1x A10G (24GB) | 16 | 64GB | ~$2.03 |

*Prices vary by region. Check [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/) for current rates.

**Recommendation**: Start with `ml.g5.2xlarge` for good balance of performance and cost.

## Performance Characteristics

| Document Type | Processing Time | Notes |
|---------------|----------------|-------|
| Single image | 2-5 seconds | Depends on resolution and complexity |
| Small PDF (1-2 pages) | 5-10 seconds | Sequential page processing |
| Handwritten text | 3-6 seconds | Accuracy depends on handwriting clarity |

**Timeout Considerations:**
- Real-time endpoints: 60-second limit
- For PDFs with 3+ pages: Process pages individually to avoid timeout
- Model loading (first request): 3-5 minutes (subsequent requests are fast)

## Project Structure

```
DeepSeek OCR/
├── container/
│   ├── Dockerfile              # Container definition
│   ├── app/
│   │   └── server.py           # FastAPI server with Transformers
│   └── requirements.txt        # Dependency documentation
├── notebooks/
│   └── deepseek_ocr_demo.ipynb # Deployment and usage demo
├── scripts/
│   ├── run_codebuild_simple.sh # Build with CodeBuild
│   └── create_source_zip.py    # Package for CodeBuild
├── buildspec.yml               # CodeBuild configuration
└── README.md                   # This file
```

## Docker Image Details

### Base Image
```dockerfile
FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04
```

Uses NVIDIA NGC registry (no Docker Hub rate limits) with CUDA 12.1 development tools for compiling flash-attention.

### Key Dependencies
- **PyTorch** 2.5.1 with CUDA 12.1
- **Transformers** 4.46.3 (official DeepSeek-OCR backend)
- **flash-attn** 2.7.3 (required for model performance)
- **FastAPI** + **uvicorn** (web server)
- **pypdfium2** (PDF processing)
- **boto3** (S3 access)
- **hf-transfer** (fast model downloads)

### Build Time
- **Total**: 15-20 minutes
- Most time spent compiling flash-attention (~8-10 minutes)

## Troubleshooting

### Endpoint Creation Fails

**Check CloudWatch Logs:**
```bash
# Find log group
aws logs describe-log-groups --log-group-name-prefix "/aws/sagemaker/Endpoints"

# View recent logs
aws logs tail /aws/sagemaker/Endpoints/your-endpoint-name --follow
```

**Common Issues:**
- Model download timeout: Increase `HF_HUB_ENABLE_HF_TRANSFER=1` is set
- OOM errors: Use larger instance (ml.g5.4xlarge)
- Image pull errors: Verify ECR permissions

### /ping Returns 503

**Cause**: Model still loading from HuggingFace (~8GB download + GPU loading)

**Solution**: Wait 3-5 minutes after endpoint shows "InService". Check CloudWatch logs for "✓ MODEL READY!"

### PDF Timeout Errors

**Problem**: Large PDFs timeout after 60 seconds

**Solutions:**
1. **Process pages individually** (recommended - see notebook example)
2. **Use SageMaker Async Inference** (requires additional S3 setup)
3. **Split PDF before processing**

### Inference Returns None

**Cause**: Missing `eval_mode=True` parameter in older versions

**Solution**: Ensure you're using the latest image. The parameter is set correctly in `container/app/server.py:177`

## Development Notes

### Testing Locally

```bash
# Build image
docker build -t deepseek-ocr:test -f container/Dockerfile .

# Run locally
docker run -p 8080:8080 \
    -e MODEL_ID=deepseek-ai/DeepSeek-OCR \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    deepseek-ocr:test

# Test endpoint
curl -X POST http://localhost:8080/invocations \
    -H "Content-Type: application/json" \
    -d '{"prompt":"<image>\\nFree OCR.","image_url":"https://example.com/test.jpg"}'
```

### Modifying the Container

1. Edit `container/app/server.py` to change inference logic
2. Edit `container/Dockerfile` to add dependencies
3. Rebuild and push: `./scripts/run_codebuild_simple.sh`
4. Update SageMaker endpoint to use new image

### Adding Custom Preprocessing

Modify the `infer_single_image()` function in `server.py` to add custom image preprocessing, prompt engineering, or post-processing logic.

## Resources

- **DeepSeek-OCR Model**: [HuggingFace Model Card](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **SageMaker BYOC Guide**: [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html)
- **SageMaker Endpoints**: [Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)
- **PyTorch Documentation**: [pytorch.org](https://pytorch.org/docs/)
- **Transformers Library**: [HuggingFace Docs](https://huggingface.co/docs/transformers/)

## License

This implementation follows the DeepSeek-OCR model license (MIT). See the [model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR) for details.

## Contributing

Issues and pull requests welcome! This is a sample implementation for AWS blog posts and can be adapted for production use cases.

## Support

For issues specific to:
- **DeepSeek-OCR model**: See the [HuggingFace model repository](https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions)
- **AWS SageMaker**: Contact AWS Support or see [AWS Forums](https://repost.aws/)
- **This implementation**: Open an issue in this repository
