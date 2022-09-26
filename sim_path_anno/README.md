# Simulated Annotations

In our project, we do not have a subject matter expert. Given resources, we can develop an interface where a pathologist uses an interface to provide annotations for each slide of WSIs. We can then use these annotations to train a model to predict the annotations for the rest of the slides. These annotations can further be enhanced to build a explainable component of the result.

For the purpose of this experiment, we came up with annotations for both mitotic and non-mitotic cells. We then randomly applied these annotations to our images and generated a dataset. We then used this dataset to train a model to predict the annotations for the rest of the images.
