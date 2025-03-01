Hello! This is Aiden and Neels final project.

Access our web app: (my google account ran out of money, so this is not currently up)
http://34.56.51.151:30001


The data is updated every 4 hours through a Cronjob that reads from the Kafka stream and updates an AWS MYSQL database.
Both the web app and Cronjob are hosted on the Google Cloud. 

I ran the code by:
  --building the images (flask app, kafka-streaming-app) to the container registry on the google cloud: 
    - docker build -t gcr.io/<your-project-id>/<name-of-container> .
    - docker push gcr.io/<your-project-id>/<name-of-container> .

 --creating a kubernetes cluster 
    - gcloud container clusters create-auto flask-kafka-cluster \
    --region us-central1 \
    --project <google-cloud-project>

    - gcloud container clusters get-credentials flask-kafka-cluster --region us-central1 --project <google-cloud-project>

 --applying the yaml files to the cluster:
    -kubectl apply -f kubernetes/



File system:
 - Flask_container: 
     -- has the flask app (app.py, static, templates)
     -- src folder has the random forest model and indexers, necessary for predicting anomalies
     -- Dockerfile and requirements.txt for containerization
     --.env has the database credentials (dont look! haha)
 - update_Events:
     --fetch_healthEvents.py: runs as a Cronjob on the Kubernetes cluster every 4 hours. 
       It takes all the newest messages from the stream and commits them to the database on AWS.
     -- Dockerfile, requirements, and .env for containerization and connecting to the db.
 - kubernetes:
    --Has the yaml files for putting both the flask-app and the kafka-consumer on the Google Cloud (I have 40 dollars leftover from a class on GCP)
 - testing:
    -- running count_entries tells you how many total entries are in the stream.

Description of work:

First, I made fetch_healthEvents. It connects to my AWS database (which I had made for another class, I just added a new table), reads all
unseen data from the stream, and adds them to the database. After adding all new messages, it prints the total messages and then exits.
Next, I made a flask app that queries the database for a specific time and place, and then separates the data into buckets and graphs it 
using plotly. It also takes my trained model from the last homework, a random forest model with .88 accuracy, and predicts whether each
data point is an anomaly. This is another option to display in the graph. I realized that it got this accuracy by always predicting
that a hospital admission is an anomaly. This is because in the small dataset every anomaly was a hospitalization. 
I then added a front-end (index.html) and some styling for it. Finally, I containerized the applications, and then hosted them through 
kubernetes on the google cloud.

Points:

Documentation (20) +
Kubernetes (25) +
AWS MYSQL database (15) + 
SparkML trained model (10) + 
Model offered in a container (10) + 
Zero-Downtime model updates (10) +    (The flask app predicts with the model each time it reads data. So we could update it with zero downtime)
Make a prediction from the Mode (5) +
Web-based graphics (10) +
Real-time event visualization (10) +  (The database is updated every 4 hours. This update is shown instantly on the Flask app)
Additional graphs (5) +
Automatic restart on failure (10)  (When the kubernetes clusters fail they are automatically restarted and rebuild the containers)

= 130 points


Split of work:

Aiden and Neel worked on this together an equal amount. Aiden pushed as we configured the google cloud stuff for his project.
