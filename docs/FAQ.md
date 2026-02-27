### FAQ 



# how does training, verification , proving all happen i.e the orchestration flow?
Please refer to the Orchestration.md file in the repo root to know this.




# in this repo, we have proved Algebraic SGD not Arithemtic SGD.

- Algebraic sgd is : 

- Arithmetic sgd is:

- Arithmetic sgd requires


# whats the flow of proof construction 
Single-step PoT
Bound PoT (on-chain dataset/policy/seed)
Recursive PoT over many local steps
Multi-round recursion across FL rounds
Then Proof of Aggregation on top of those round-level results


# what is the input problem that we have to solve? - Ojas and Sameer


### Quoting copilot : 
- Where do the actual values (activations, grads, updates) come from?
They’re assumed to come from your ML framework / training run, not from ONNX:

- The external trainer runs a step: computes forward activations, upstream gradients, weight gradients, and updated weights.
- Those tensors are fed as witnesses to the ZK training graph (forward, backward, optimizer graphs in training.rs).
- Inside the circuit, the gadgets recompute what they should be (via the constraints above) and enforce equality.
- So the proof says: “there exist x, W_t, activations, grads, W_{t+1} such that all these relations hold,” not “I extracted the backward from ONNX.”


- basically we have to figure out where we are getting the actual values/code of the model from ? (backward pass and update steps.)

- outcome : exact idea about how inputs flow throough the project.
- delivarable : make a data_flow.md that tracks flow of the data throughout the project.


- references : training_compile.rs line 76

# how do we solve the data integration into proof. - Abhishek / Sameer
- data binding happens first, before training.
- the client has to commit to a dataset before he starts his training.
- the final verifier has to be able to check that the client has trained on the correct dataset 
with the agreed upon model.
- can be solved through merkle trees and merkle roots, blockchain already solves this.
- good reason to check bitcoin core.


# basic blocks writing and layers and unit tests - Arjun
- study src/basic_block and src/layer
- understand their dependence 
- make the basic blocks code modular so that we dont have a single point of failure.
- ### dont mix unit tests with actual code, write unit tests separately.
- add to the file_structure.md doc what u have understood from src/basic_block and src/layer.
- document what blocks we have right now and what do we need inorder to cover most models' backward pass code 



## update security.md doc to match architecture functionality. - Sameer - DONE.