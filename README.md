# Hugging Face AWS Inferentia2 samples

This repository contains sample notebooks and scripts to demonstrate how to deploy Hugging Face models, like BERT, GPT-2 Stable-Diffusion, T5 with Amazon SageMaker, and AWS Inferentia.2
You will learn how to: 

1. Convert your models to AWS Neuron (Inferentia")
2. Create a custom `inference.py` scripts
3. Create and upload the neuron model and inference script to Amazon S3
4. Deploy a Real-time Inference Endpoint on Amazon SageMaker
5. Run and evaluate Inference performance on Inferentia2

## 📕 Background

[AWS Inf2](https://aws.amazon.com/ec2/instance-types/inf2/?nc1=h_ls) instances are powered by [AWS Inferentia2](https://aws.amazon.com/machine-learning/inferentia/), the second-generation AWS Inferentia accelerator. 

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service for data science and machine learning (ML) workflows.
You can use Amazon SageMaker to simplify the process of building, training, and deploying ML models.

The [SageMaker example notebooks](https://sagemaker-examples.readthedocs.io/en/latest/) are Jupyter notebooks that demonstrate the usage of Amazon SageMaker.

## 🛠 Setup

The quickest setup to run example notebooks includes:
- An [AWS account](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-account.html)
- Proper [IAM User and Role](http://docs.aws.amazon.com/sagemaker/latest/dg/authentication-and-access-control.html) setup
- An [Amazon SageMaker Notebook Instance](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
- An [S3 bucket](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-config-permissions.html)


---

*If you are going to use Sagemaker in a local environment (not SageMaker Studio or Notebook Instances). You need access to an IAM Role with the required permissions for Sagemaker. You can find [here](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) more about it.*