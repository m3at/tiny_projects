GCR_NS?=CLUSTER_NAME
DOCKER_IMAGE=gcr.io/$(GCR_NS)/poc/training-image
K8S_CLUSTER?=GKE_CLUSTER_NAME

check-kubectl-context:
	@test "$$(kubectl config current-context)" = "$(K8S_CLUSTER)"

start-job: check-kubectl-context
	kubectl -n $(GCR_NS) apply -f ./job-def.yaml

check-job:
	@# Check info about specific job
	kubectl get -f job-def.yaml

# Check logs
# Assuming just one, the first:
check-log:
	@kubectl logs -f $(shell kubectl get pods --no-headers=true -o=name | head -n 1)

delete-job:
	kubectl delete -f ./job-def.yaml

# Run this to use minikube's docker: eval $(minikube docker-env)
build:
	docker build -t $(DOCKER_IMAGE) .

push:
	docker push $(DOCKER_IMAGE)

docker-run:
	docker run $(DOCKER_IMAGE)
