import requests

event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49680309744750414074628682100614380931881771554178072578",
                "data": "eyJyaWRlIjogeyJQVUxvY2F0aW9uSUQiOiAxMCwgIkRPTG9jYXRpb25JRCI6IDUwfSwgInJpZGVfaWQiOiAxMjN9",
                "approximateArrivalTimestamp": 1794931289.567
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49680309744750414074628682100614380931881771554178072578",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::786875389232:role/lambda-kinesis-role",
            "awsRegion": "eu-north-1",
            "eventSourceARN": "arn:aws:kinesis:eu-north-1:786875389232:stream/ride_events"
        }
    ]
}

url = "http://localhost:8080/2015-03-31/functions/function/invocations"
response = requests.post(url, json=event)
print(response.json())