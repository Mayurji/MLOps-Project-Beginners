## Set up Minikube On Mac and Deploy a simple App

1. Check the pre-requisites 
2. Install Kubernetes using Minikube
3. Install Kubectl
4. Start Minikube and Verify Kubernetes Cluster
5. Deploy a simple app.
6. Test and Clean up.

Prerequisite step:

1. [Docker Desktop Installation](https://docs.docker.com/desktop/setup/install/mac-install/)
2. [Minikube Installation](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Fx86-64%2Fstable%2Fbinary+download)
3. [Kubectl Installation](https://kubernetes.io/docs/tasks/tools/)
4. Commands:

```bash
minikube start
kubectl create deployment hello-minikube --image=kicbase/echo-server:1.0
kubectl expose deployment hello-minikube --type=NodePort --port=8080
kubectl get services hello-minikube
kubectl port-forward service/hello-minikube 7080:8080 #to view locally on browser
```

## Clean up:
```bash
kubectl delete service hello-minikube
kubectl delete deployment hello-minikube
minikube stop
minikube delete
```
## Run a Machine Learning Model and Serve it as FastAPI service. Bonus (Containerzation of FastAPI)

**Installation**

```
pip install fastapi uvicorn scikit-learn
```

### Train a Machine Learning Model

1. Train a randomforest model using Sklearn.
2. Evaluate the model.
3. Save the model as Pickle file.
4. Load and test the saved model

### Create FastAPI to serve the Model

1. Create a FastAPI app
2. Load the saved model from previous section
3. Create Pydantic Class for data validation.
4. Create two API ('health_check', 'predict')

### Running FastAPI Service

```
uvicorn fastapi_service:app --reload
```

Go to `http://127.0.0.1:8000/docs` to test the API

**Curl Request**

```
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "features": [5.1, 3.5, 1.4, 0.2]
}'
```

### Dockerization

```bash
docker build -t my-app:v1 .
docker run -p 8000:8000 my-app:v1
```