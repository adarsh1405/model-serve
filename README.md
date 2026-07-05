## Setup env & install requirements packages

python3 -m venv .env
source .env/bin/activate

pip3 install -r requirements.txt

deactivate (optional) // to remove all the packages install in the venv

## once done , verify the MLFLow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 7006

## Run MLFLOW in docker (optional)
- Install docker
- Use kind/k3d/minikube for local cluster setup
    'kind create cluster --name=demo-cluster'
- mlflow community helm chart
    follow - https://community-charts.github.io/docs/charts/mlflow/basic-installation
    - helm install <> & do the port-forward

## DVC 
- managed for larger datsets
- python3 -m pip install dvc
- dvc pull
- cat .dvc/config // gets you the config like where its being pulled from

## MLFLOW commnds

```
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("my-first-experiment")
```

```
with mlflow.start_run():
    mlflow.log_param("test_param", "test_value")
    print("✓ Successfully connected to MLflow!")
```


```
with mlflow.start_run(run_name=args.run) as run:
    mlflow.log_param("test_param", "test_value")
    print("✓ Successfully connected to MLflow!")
```
mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})



## 2. Predict prices

**Single house** (any values you omit fall back to sensible defaults):

```bash
python predict.py --area 7420 --bedrooms 4 --bathrooms 2 --stories 3 \
    --mainroad yes --guestroom no --basement no \
    --hotwaterheating no --airconditioning yes
```

**Batch of houses** from a CSV (must contain the feature columns, no `price` needed):

```bash
python predict.py --input new_houses.csv --output predictions.csv
```

## 3. Serve the model as a REST API (Flask)

Start the API server:

```bash
python app.py
```

The server runs at `http://127.0.0.1:5001` (set a different port with the
`PORT` env var, e.g. `PORT=8000 python app.py`).

### Endpoints

| Method & path  | Description                          |
| -------------- | ------------------------------------ |
| `GET /`        | Basic info and required fields       |
| `GET /health`  | Health check (confirms model loaded) |
| `POST /predict`| Predict price(s) from JSON           |

**Single house:**

```bash
curl -X POST http://127.0.0.1:5001/predict \
    -H "Content-Type: application/json" \
    -d '{"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
         "mainroad": "yes", "guestroom": "no", "basement": "no",
         "hotwaterheating": "no", "airconditioning": "yes"}'
# -> {"predicted_price": 7699822.2}
```

**Multiple houses** (send a JSON list):

```bash
curl -X POST http://127.0.0.1:5001/predict \
    -H "Content-Type: application/json" \
    -d '[{"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
          "mainroad": "yes", "guestroom": "no", "basement": "no",
          "hotwaterheating": "no", "airconditioning": "yes"}]'
# -> {"predicted_prices": [7699822.2]}
```

## Files

| File                         | Purpose                                        |
| ---------------------------- | ---------------------------------------------- |
| `train_model.py`             | Trains the model and saves `house_price_model.pkl` |
| `predict.py`                 | Loads the `.pkl` and predicts prices (CLI)     |
| `app.py`                     | Flask REST API serving the model at `/predict` |
| `requirements.txt`           | Python dependencies                            |
| `Housing-selected-columns.csv` | The dataset                                  |



### Deploy / Serve using Kserve

- Install cert-manager
- [Follow these steps](https://github.com/iam-veeramalla/Intent-classifier-model/blob/kserve/KServe-implementation.md)
- Create namespace , CRDs , contoller 
- Apply the Infernece service --> this will pull models from a downloadable link
- modelFormat (sklearn , tensorflow , pytorch)
- NOTE/ADVICE : We can keep the model (.pkl) file in the github repo by releasing the artifact (which will give us a downloadable link)


