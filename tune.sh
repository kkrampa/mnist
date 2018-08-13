#!/bin/bash
USERNAME=$1
PROJECT_NAME=$2

lr="1e0"
hidden="8"
just create job single --project $USERNAME/$PROJECT_NAME --module mnist --learning_rate $lr --hidden1 $hidden --hidden2 $hidden --name lr-$lr-hidden-$hidden --instance-type c4.2xlarge --description "learning rate : $lr and number of hidden layer nodes =$hidden" --python-version 3 --time-limit 1h
just start job $PROJECT_NAME/lr-$lr-hidden-$hidden
echo "Job lr-$lr-hidden-$hidden started!"

lr="1e-1"
hidden="8"
just create job single --project $USERNAME/$PROJECT_NAME --module mnist --learning_rate $lr --hidden1 $hidden --hidden2 $hidden --name lr-$lr-hidden-$hidden --instance-type c4.2xlarge --description "learning rate : $lr and number of hidden layer nodes =$hidden" --python-version 3 --time-limit 1h
just start job $PROJECT_NAME/lr-$lr-hidden-$hidden
echo "Job lr-$lr-hidden-$hidden started!"

lr="1e-2"
hidden="8"
just create job single --project $USERNAME/$PROJECT_NAME --module mnist --learning_rate $lr --hidden1 $hidden --hidden2 $hidden --name lr-$lr-hidden-$hidden --instance-type c4.2xlarge --description "learning rate : $lr and number of hidden layer nodes =$hidden" --python-version 3 --time-limit 1h
just start job $PROJECT_NAME/lr-$lr-hidden-$hidden
echo "Job lr-$lr-hidden-$hidden started!"

lr="1e0"
hidden="64"
just create job single --project $USERNAME/$PROJECT_NAME --module mnist --learning_rate $lr --hidden1 $hidden --hidden2 $hidden --name lr-$lr-hidden-$hidden --instance-type c4.2xlarge --description "learning rate : $lr and number of hidden layer nodes =$hidden" --python-version 3 --time-limit 1h
just start job $PROJECT_NAME/lr-$lr-hidden-$hidden
echo "Job lr-$lr-hidden-$hidden started!"

lr="1e-1"
hidden="64"
just create job single --project $USERNAME/$PROJECT_NAME --module mnist --learning_rate $lr --hidden1 $hidden --hidden2 $hidden --name lr-$lr-hidden-$hidden --instance-type c4.2xlarge --description "learning rate : $lr and number of hidden layer nodes =$hidden" --python-version 3 --time-limit 1h
just start job $PROJECT_NAME/lr-$lr-hidden-$hidden
echo "Job lr-$lr-hidden-$hidden started!"

lr="1e-2"
hidden="64"
just create job single --project $USERNAME/$PROJECT_NAME --module mnist --learning_rate $lr --hidden1 $hidden --hidden2 $hidden --name lr-$lr-hidden-$hidden --instance-type c4.2xlarge --description "learning rate : $lr and number of hidden layer nodes =$hidden" --python-version 3 --time-limit 1h
just start job $PROJECT_NAME/lr-$lr-hidden-$hidden
echo "Job lr-$lr-hidden-$hidden started!"

