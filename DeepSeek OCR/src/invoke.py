import argparse, json, boto3, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--image-url")
    ap.add_argument("--pdf-url")
    ap.add_argument("--prompt", default="<image>\n<|grounding|>Convert the document to markdown.")
    args = ap.parse_args()

    rt = boto3.client("sagemaker-runtime")
    body = {"prompt": args.prompt}
    if args.image_url: body["image_url"] = args.image_url
    if args.pdf_url: body["pdf_url"] = args.pdf_url

    resp = rt.invoke_endpoint(EndpointName=args.endpoint, ContentType="application/json", Body=json.dumps(body))
    print(resp["Body"].read().decode("utf-8"))

if __name__ == "__main__":
    main()
