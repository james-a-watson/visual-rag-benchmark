# ColPali API for Google Colab
# Install required packages first by running this in a cell:
# !pip install fastapi uvicorn colpali-engine torch torchvision pillow pyngrok

# IMPORTANT: Set environment variables BEFORE importing torch
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Now import torch and other packages
import torch
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional
from PIL import Image
import io
import time
import secrets
import asyncio
import nest_asyncio
from colpali_engine.models import ColPali, ColPaliProcessor
from pyngrok import ngrok
import uvicorn
from threading import Thread

# Allow nested event loops for Colab
nest_asyncio.apply()

# Set torch threading (with error handling)
try:
    torch.set_num_threads(2)
except RuntimeError:
    print("⚠️  Could not set torch num_threads (already initialized)")

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    print("⚠️  Could not set torch interop_threads (already initialized)")

# Authentication setup
API_KEY = os.getenv("COLPALI_API_KEY", "AliceInWonderland")
security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the API key from Authorization header"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def verify_api_key_header(x_api_key: str = Header(None)):
    """Verify API key from X-API-Key header"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


app = FastAPI(
    title="ColPali API - Colab Edition",
    description="Document Retrieval Vision-Language Model API for Google Colab",
)

# Global variables to store model and processor
model = None
processor = None
model_loaded = False


def load_model():
    """Load the ColPali model - called once during startup"""
    global model, processor, model_loaded

    if model_loaded:
        return

    print("🔄 Loading ColPali model... This may take 5-10 minutes on first run.")

    model_name = "vidore/colpali-v1.3"

    # Check if GPU is available in Colab
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.bfloat16
        print("🚀 Using GPU acceleration!")
    else:
        device = "cpu"
        torch_dtype = (
            torch.float32
        )  # Changed from float16 to float32 for CPU compatibility
        print("⚠️  Using CPU - consider enabling GPU in Colab Runtime settings")

    try:
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        ).eval()

        processor = ColPaliProcessor.from_pretrained(model_name)
        model_loaded = True
        print("✅ ColPali model loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("This might be due to network or memory constraints.")
        raise e


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.post("/search")
async def search_documents(
    images: List[UploadFile] = File(...),
    queries: List[str] = ["What is in this document?"],
    api_key: str = Depends(verify_api_key),
):
    """
    Search through document images using text queries
    Returns similarity scores between queries and images
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        print(f"Processing {len(images)} images with {len(queries)} queries")

        # Process uploaded images
        pil_images = []
        for img_file in images:
            image_data = await img_file.read()
            pil_image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if necessary
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_images.append(pil_image)

        # Process inputs using ColPali processor
        start_time = time.time()

        batch_images = processor.process_images(pil_images).to(model.device)
        batch_queries = processor.process_queries(queries).to(model.device)

        # Forward pass
        with torch.no_grad():
            image_embeddings = model(**batch_images)
            query_embeddings = model(**batch_queries)

        # Calculate similarity scores
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)

        inference_time = time.time() - start_time
        print(f"✅ Inference completed in {inference_time:.2f} seconds")

        # Format results
        results = []
        for i, query in enumerate(queries):
            query_scores = []
            for j, img_file in enumerate(images):
                query_scores.append(
                    {
                        "image_filename": img_file.filename,
                        "similarity_score": float(scores[i][j]),
                    }
                )

            # Sort by similarity score (highest first)
            query_scores.sort(key=lambda x: x["similarity_score"], reverse=True)
            results.append({"query": query, "matches": query_scores})

        return {
            "results": results,
            "inference_time": f"{inference_time:.2f}s",
            "num_images": len(images),
            "num_queries": len(queries),
        }

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/single_query")
async def single_query(
    image: UploadFile = File(...),
    query: str = "What is in this document?",
    api_key: str = Depends(verify_api_key),
):
    """Simple single image, single query endpoint"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        print(f"Processing single query: '{query}'")

        # Process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Process inputs
        start_time = time.time()

        batch_images = processor.process_images([pil_image]).to(model.device)
        batch_queries = processor.process_queries([query]).to(model.device)

        # Forward pass
        with torch.no_grad():
            image_embeddings = model(**batch_images)
            query_embeddings = model(**batch_queries)

        scores = processor.score_multi_vector(query_embeddings, image_embeddings)
        inference_time = time.time() - start_time

        print(f"✅ Single query completed in {inference_time:.2f} seconds")

        return {
            "query": query,
            "similarity_score": float(scores[0][0]),
            "inference_time": f"{inference_time:.2f}s",
            "image_filename": image.filename,
        }

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed_image")
async def embed_image(
    image: UploadFile = File(...), api_key: str = Depends(verify_api_key)
):
    """Get embeddings for a single image"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        print(f"Generating embeddings for image: {image.filename}")

        # Process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Process inputs
        start_time = time.time()
        batch_images = processor.process_images([pil_image])

        # Move to device but preserve dtypes for different tensor types
        batch_images_device = {}
        for key, value in batch_images.items():
            if torch.is_tensor(value):
                if value.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    # Keep integer dtypes as-is (for indices/attention masks)
                    batch_images_device[key] = value.to(model.device)
                else:
                    # Convert floating point tensors to model dtype
                    batch_images_device[key] = value.to(
                        model.device, dtype=next(model.parameters()).dtype
                    )
            else:
                batch_images_device[key] = value

        # Forward pass
        with torch.no_grad():
            image_embeddings = model(**batch_images_device)

        inference_time = time.time() - start_time

        # Convert embeddings to list for JSON serialization
        embeddings_list = image_embeddings[0].cpu().float().numpy().tolist()

        print(f"✅ Image embedding completed in {inference_time:.2f} seconds")

        return {
            "embeddings": embeddings_list,
            "embedding_shape": list(image_embeddings[0].shape),
            "embedding_dimension": image_embeddings[0].shape[-1],
            "num_patches": image_embeddings[0].shape[0],
            "inference_time": f"{inference_time:.2f}s",
            "image_filename": image.filename,
        }

    except Exception as e:
        print(f"❌ Error during image embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import Request
import pickle


@app.post("/similarity_score")
async def similarity_score(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    try:
        print("Request recieved.")
        # Get raw body and unpickle
        body = await request.body()
        print("body retrieved")
        data = pickle.loads(body)
        print("Data unpickled.")

        query_embeddings = data["query_embeddings"]
        image_embeddings = data["image_embeddings"]
        # Rest of your original function code...
        print(
            f"Calculating similarity score for {len(query_embeddings)} queries and {len(image_embeddings)} images"
        )
        query_tensor = torch.tensor(query_embeddings).to(model.device)
        image_tensor = torch.tensor(image_embeddings).to(model.device)

        start_time = time.time()
        scores = processor.score_multi_vector(query_tensor, image_tensor)
        inference_time = time.time() - start_time
        print(f"✅ Scoring completed in {inference_time:.2f} seconds")
        return {
            "similarity_scores": scores[0].cpu().float().numpy().tolist(),
        }

    except Exception as e:
        print(f"❌ Error during text embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/server_side_similarity_score")
async def server_side_similarity_score(
    query: str,
    api_key: str = Depends(verify_api_key),
):
    try:
        print("Request recieved.")
        # body = await request.body()
        # data = pickle.loads(body)
        # query_embeddings = data["query_embeddings"]

        print(f"Generating embeddings for query: '{query}'")
        start_time = time.time()

        batch_queries = processor.process_queries([query])
        # Handle device and dtype conversion properly
        batch_queries_device = {}
        for key, value in batch_queries.items():
            if torch.is_tensor(value):
                if value.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    batch_queries_device[key] = value.to(model.device)
                else:
                    batch_queries_device[key] = value.to(
                        model.device, dtype=next(model.parameters()).dtype
                    )
            else:
                batch_queries_device[key] = value

        batch_queries = batch_queries_device

        # Forward pass
        with torch.no_grad():
            query_embeddings = model(**batch_queries)

        inference_time = time.time() - start_time
        print(f"✅ Text embedding completed in {inference_time:.2f} seconds")

        query_tensor = (
            torch.tensor(query_embeddings).float().to(model.device)
        )  # Convert to float32
        image_tensor = (
            torch.tensor(image_embeddings).float().to(model.device)
        )  # Convert to float32
        print(
            f"Calculating similarity score for {len(query_embeddings)} queries and {len(image_embeddings)} images"
        )
        start_time = time.time()
        scores = processor.score_multi_vector(query_tensor, image_tensor)
        inference_time = time.time() - start_time
        print(f"✅ Scoring completed in {inference_time:.2f} seconds")

        return {
            "similarity_scores": scores[0].cpu().float().numpy().tolist(),
        }

    except Exception as e:
        print(f"❌ Error during similarity score embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed_text")
async def embed_text(query: str, api_key: str = Depends(verify_api_key)):
    """Get embeddings for text query"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        print(f"Generating embeddings for query: '{query}'")
        start_time = time.time()

        batch_queries = processor.process_queries([query])
        # Handle device and dtype conversion properly
        batch_queries_device = {}
        for key, value in batch_queries.items():
            if torch.is_tensor(value):
                if value.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    batch_queries_device[key] = value.to(model.device)
                else:
                    batch_queries_device[key] = value.to(
                        model.device, dtype=next(model.parameters()).dtype
                    )
            else:
                batch_queries_device[key] = value

        batch_queries = batch_queries_device

        # Forward pass
        with torch.no_grad():
            query_embeddings = model(**batch_queries)

        inference_time = time.time() - start_time

        # Convert embeddings to list for JSON serialization
        embeddings_list = query_embeddings[0].cpu().float().numpy().tolist()

        print(f"✅ Text embedding completed in {inference_time:.2f} seconds")

        return {
            "embeddings": embeddings_list,
            "embedding_shape": list(query_embeddings[0].shape),
            "embedding_dimension": query_embeddings[0].shape[-1],
            "num_tokens": query_embeddings[0].shape[0],
            "inference_time": f"{inference_time:.2f}s",
            "query": query,
        }

    except Exception as e:
        print(f"❌ Error during text embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    device_info = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "healthy" if model_loaded else "loading",
        "device": device_info,
        "model_loaded": model_loaded,
        "gpu_available": torch.cuda.is_available(),
        "model": "vidore/colpali-v1.3" if model_loaded else "not loaded",
    }


@app.get("/")
async def root():
    return {
        "message": "ColPali API is running on Google Colab!",
        "model_loaded": model_loaded,
        "endpoints": {
            "/search": "Multi-image, multi-query document search",
            "/single_query": "Single image, single query",
            "/embed_image": "Get image embeddings",
            "/embed_text": "Get text embeddings",
            "/health": "Health check",
        },
    }


def setup_ngrok(auth_token=None):
    """Setup ngrok tunnel for public access"""
    try:
        if auth_token:
            ngrok.set_auth_token(auth_token)

        # Kill any existing tunnels first
        ngrok.kill()

        # Create tunnel
        public_url = ngrok.connect(8000)
        print(f"🌐 Public URL: {public_url}")
        print(f"📄 API docs available at: {public_url}/docs")
        return public_url
    except Exception as e:
        print(f"❌ Failed to create ngrok tunnel: {e}")
        print("💡 You can still use the API within Colab at http://localhost:8000")
        return None


def run_server():
    """Run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# Main function to start everything
def start_colpali_api(ngrok_auth_token=None, use_ngrok=True):
    """
    Start the ColPali API server in Google Colab

    Args:
        ngrok_auth_token: Your ngrok auth token (optional, for persistent URLs)
        use_ngrok: Whether to create a public ngrok tunnel
    """
    global API_KEY

    # Generate API key if using default
    if API_KEY == "AliceInWonderland":
        generated_key = secrets.token_urlsafe(12)
        print(f"🔑 Generated API Key: {generated_key}")
        print("⚠️  SAVE THIS KEY! You'll need it to authenticate API requests")
        API_KEY = generated_key

    print("🚀 Starting ColPali API server on Google Colab...")
    print("📄 ColPali specializes in document understanding and retrieval")
    print("🔐 Authentication required - use Bearer token or X-API-Key header")

    if use_ngrok:
        try:
            print("🔄 Setting up ngrok tunnel...")
            public_url = setup_ngrok(ngrok_auth_token)
            if public_url:
                print(f"✅ Tunnel ready! Access from your laptop: {public_url}")
        except Exception as e:
            print(f"❌ Tunnel setup failed: {e}")
            print("🔄 Starting server without tunnel (Colab-only access)")

    # Start server
    print("🔄 Server starting...")
    run_server()


# Alternative function to create tunnel after server is running
def create_tunnel_after_startup(auth_token=None):
    """Call this in a separate cell after the server starts"""
    try:
        if auth_token:
            ngrok.set_auth_token(auth_token)

        # Kill existing tunnels
        ngrok.kill()

        # Create new tunnel
        public_url = ngrok.connect(8000)
        print(f"🌐 Tunnel created: {public_url}")
        print(f"📄 API docs: {public_url}/docs")
        print(f"🧪 Test health: {public_url}/health")
        return public_url
    except Exception as e:
        print(f"❌ Failed to create tunnel: {e}")
        return None


# Alternative setup function that handles torch initialization better
def setup_torch_environment():
    """Call this BEFORE importing torch if you get thread errors"""
    import os

    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    print("✅ Environment variables set for optimal torch performance")


# Usage instructions for Colab
print(
    """
🚀 ColPali API for Google Colab Setup Instructions:

1. First, install dependencies:
   !pip install fastapi uvicorn colpali-engine torch torchvision pillow pyngrok

2. (Optional) Enable GPU in Runtime > Change runtime type > Hardware accelerator > GPU

3. If you get torch thread errors, restart runtime and run setup_torch_environment() first

4. Start the API server:
   start_colpali_api()

5. For persistent public URLs, get a free ngrok token at https://ngrok.com and use:
   start_colpali_api(ngrok_auth_token="your_token_here")

6. Use the generated API key to authenticate your requests!

Example usage:
   start_colpali_api()

If you get threading errors:
   1. Runtime > Restart runtime
   2. setup_torch_environment()  # Run this first
   3. Then run the main code
"""
)
