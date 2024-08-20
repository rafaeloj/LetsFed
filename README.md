## Client config
```
  [
      {
          "cid": "",
          "dataset": "",
          "epochs": "",
          "isParticipate": "",
          "dirichlet_alpha": "",
          "exploitation": "",
          "exploration": "",
          "threshold": "",
          "model_type": "",
          "train_p": "dirichletpartitioner",
          "test_p": "iidpartitioner",
          "drivers": [
              {
                  "did": "accuracy_driver",
                  "willing_perc": 1.0
              },
              {
                  "did": "curiosity_driver",
                  "curiosity": 0.1,
                  "willing_perc": 0.80
              }
          ]
      }
  ]
```