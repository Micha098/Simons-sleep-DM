#!/usr/bin/env python
import os
import subprocess

os.environ['ACCESS_URL'] = "s3://empatica-us-east-1-prod-data/v2/058/"
os.environ['LOCAL_PATH'] = "/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/"
os.environ['AWS_ACCESS_KEY_ID'] = "AKIAWWZYTIF5SUNKK65K"
os.environ['AWS_SECRET_ACCESS_KEY'] = "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"

sync_command = f"aws s3 sync {os.environ['ACCESS_URL']} {os.environ['LOCAL_PATH']}"
subprocess.run(sync_command, shell=True)
subprocess.run(f"{sync_command} > output.txt", shell=True)
