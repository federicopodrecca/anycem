from abc import abstractmethod
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from anycem.extra_nn import ConceptEmbedding

class NeuralNet(pl.LightningModule):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__()
        self.input_features = input_features
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = BCELoss(reduction="mean")
        if n_classes == 1:
            self.task_loss = BCEWithLogitsLoss(reduction="mean")
        else:
            self.task_loss = CrossEntropyLoss(reduction="mean")
        self.nnl = torch.nn.NLLLoss(reduction="mean")

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def _unpack_input(self, I):
        raise NotImplementedError

    def training_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)

        y_preds = self.forward(X) # cosa fa

        loss = self.task_loss(y_preds.squeeze(), y_true.float().squeeze())
        task_accuracy = roc_auc_score(y_true.squeeze(), y_preds.detach())
        self.log("train_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)
        y_preds = self.forward(X)
        loss = self.task_loss(y_preds.squeeze(), y_true.float().squeeze())
        task_accuracy = roc_auc_score(y_true.squeeze(), y_preds.detach())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)
        y_preds = self.forward(X)
        loss = self.task_loss(y_preds.squeeze(), y_true.float().squeeze())
        task_accuracy = roc_auc_score(y_true.squeeze(), y_preds.detach())
        self.log("test_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

class GenerativeCEM(NeuralNet):
    def __init__(self, input_features: int, n_concepts: int, emb_size: int,
                 learning_rate: float = 0.001, w_concept_loss: float = 1.0, w_loss: float = 0.2, w_concept_loss_int: float = 1.0,
                concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1, active_intervention_values=None, inactive_intervention_values=None,
                 intervention_idxs=None, device_gpu = False):
        super().__init__(input_features, 0, emb_size, learning_rate)
        self.n_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight
        self.w_concept_loss = w_concept_loss  # <-- Salvataggio pesi
        self.w_loss = w_loss
        self.w_concept_loss_int = w_concept_loss_int
        self.device_gpu = device_gpu
        self.encoder = ConceptEmbedding(input_features, n_concepts, emb_size, 
                                        active_intervention_values, inactive_intervention_values, intervention_idxs) # da controllare per intervention 
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_concepts * emb_size, emb_size), # cambaire n_conp * emb, Seq combina sequenza di layers
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, input_features),
            torch.nn.LeakyReLU()
        )
        self.mse = torch.nn.MSELoss()

    def _unpack_input(self, I):
        return I[0], I[1]

    def forward(self, X, c_true = None, intervention_idx = None, explain=False): # int_concept, indx_concept anche forward cbm
        c_emb, c_pred = self.encoder(X, intervention_idxs = intervention_idx, c = c_true) # passare nuovi input
        shape_c_emb = c_emb.shape
        c_emb = c_emb.reshape(shape_c_emb[0], shape_c_emb[1] * shape_c_emb[2])
        # c_pred[:, :-2] = torch.softmax(c_pred[:, :-2], dim=1)
        # c_pred[:, -2:] = torch.softmax(c_pred[:, -2:], dim=1)
        return self.decoder(c_emb), c_pred

    def training_step(self, I, batch_idx):
        X, c_true = self._unpack_input(I)

        rec_emb, c_preds = self.forward(X)
        random_int = random.randint(0, 2)
        random_concept = torch.tensor([[random.randint(0,1), random.randint(0,1), random.randint(0,1)]], device = X.device)
        random_concept = random_concept.repeat(X.shape[0], 1)
        rec_emb_int, cp_int = self.forward(X, intervention_idx = [random_int], c_true = random_concept)
        rec_ei, c_preds_int = self.forward(rec_emb_int) 
        concept_loss_int = self.bce(c_preds_int[:, random_int], random_concept[:, random_int].float())
        concept_loss = self.bce(c_preds, c_true.float())
        # concept_loss_1 = self.nnl(torch.log(c_preds[:, :-2]), c_true[:, :-2].float().argmax(dim=-1))
        # concept_loss_2 = self.nnl(torch.log(c_preds[:, -2:]), c_true[:, -2:].float().argmax(dim=-1))
        mse_loss = self.mse(rec_emb, X)
        base_loss = concept_loss + mse_loss + concept_loss_int
        loss = self.w_concept_loss + concept_loss + self.w_loss * mse_loss + self.w_concept_loss_int * concept_loss_int

        if self.device_gpu:
            concept_accuracy = roc_auc_score(c_true.cpu().numpy(), c_preds.detach().cpu().numpy())
        else:
            concept_accuracy = roc_auc_score(c_true, c_preds.detach())
        
        self.log("mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, I, batch_idx):
        X, c_true = self._unpack_input(I)

        rec_emb, c_preds = self.forward(X)
        random_int = random.randint(0, 2)
        random_concept = torch.tensor([[random.randint(0,1), random.randint(0,1), random.randint(0,1)]], device = X.device)
        random_concept = random_concept.repeat(X.shape[0], 1)
        rec_emb_int, cp_int = self.forward(X, intervention_idx = [random_int], c_true = random_concept)
        rec_ei, c_preds_int = self.forward(rec_emb_int) 
        concept_loss_int = self.bce(c_preds_int[:, random_int], random_concept[:, random_int].float())

        concept_loss = self.bce(c_preds, c_true.float())
        # concept_loss_1 = self.nnl(torch.log(c_preds[:, :-2]), c_true[:, :-2].float().argmax(dim=-1))
        # concept_loss_2 = self.nnl(torch.log(c_preds[:, -2:]), c_true[:, -2:].float().argmax(dim=-1))
        mse_loss = self.mse(rec_emb, X)
        base_loss = concept_loss + mse_loss + concept_loss_int
        loss = self.w_concept_loss + concept_loss + self.w_loss * mse_loss + self.w_concept_loss_int * concept_loss_int

        if self.device_gpu:
            concept_accuracy = roc_auc_score(c_true.cpu().numpy(), c_preds.detach().cpu().numpy())
        else:
            concept_accuracy = roc_auc_score(c_true, c_preds.detach())


        self.log("val_mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, I, batch_idx):
        X, c_true = self._unpack_input(I)

        rec_emb, c_preds = self.forward(X)
        random_int = random.randint(0, 2)
        random_concept = torch.tensor([[random.randint(0,1), random.randint(0,1), random.randint(0,1)]], device = X.device)
        random_concept = random_concept.repeat(X.shape[0], 1)
        rec_emb_int, cp_int = self.forward(X, intervention_idx = [random_int], c_true = random_concept)
        rec_ei, c_preds_int = self.forward(rec_emb_int) 
        concept_loss_int = self.bce(c_preds_int[:, random_int], random_concept[:, random_int].float())

        concept_loss = self.bce(c_preds, c_true.float())
        # concept_loss_1 = self.nnl(torch.log(c_preds[:, :-2]), c_true[:, :-2].float().argmax(dim=-1))
        # concept_loss_2 = self.nnl(torch.log(c_preds[:, -2:]), c_true[:, -2:].float().argmax(dim=-1))
        mse_loss = self.mse(rec_emb, X)
        base_loss = concept_loss + mse_loss + concept_loss_int
        loss = self.w_concept_loss + concept_loss + self.w_loss * mse_loss + self.w_concept_loss_int * concept_loss_int

        if self.device_gpu:
            concept_accuracy = roc_auc_score(c_true.cpu().numpy(), c_preds.detach().cpu().numpy())
        else:
            concept_accuracy = roc_auc_score(c_true, c_preds.detach())

        self.log("test_mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_base_loss", base_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
class CBM(NeuralNet):
    def __init__(self, input_features: int, n_concepts: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None):
        super().__init__(input_features, 0, emb_size, learning_rate)
        self.n_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names

        # Encoder che mappa l'input in uno spazio latente
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        
        # Predittore dei concetti
        self.concept_predictor = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_concepts),
            torch.nn.Sigmoid()  # Predizione dei concetti
        )
        
        # Decoder che mappa l'embedding nei concetti e li ricostruisce
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, input_features),  # Ricostruzione dell'input
        )

        # Perdita MSE per la ricostruzione
        self.mse = torch.nn.MSELoss()

    def _unpack_input(self, I):
        return I[0], I[1]  # Separazione tra input e concetti (task è rimosso)

    def forward(self, X):
        # Predizione dei concetti
        c_pred = self.concept_predictor(self.encoder(X)) # sostituisco a c_pred agli indx di interesse quelli intervenuti
        # Ricostruzione tramite decoder
        rec_emb = self.decoder(c_pred)
        return rec_emb, c_pred  # Ritorna la ricostruzione dell'input e i concetti

    def training_step(self, I, batch_idx):
        X, c_true = self._unpack_input(I)

        # Predizione concetti e ricostruzione
        rec_emb, c_preds = self.forward(X)

        # Perdita per la previsione dei concetti (BCE)
        concept_loss = self.bce(c_preds, c_true.float())

        # Perdita per la ricostruzione dell'input (MSE)
        mse_loss = self.mse(rec_emb, X)
        
        # La perdita totale ora è solo la combinazione tra concetti e ricostruzione
        loss =  concept_loss + 0.1 * mse_loss
        self.log("concept_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log delle metriche
        concept_accuracy = roc_auc_score(c_true.ravel(), c_preds.detach().ravel())
        

        self.log("concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, I, batch_idx):
        X, c_true = self._unpack_input(I)

        # Predizione concetti e ricostruzione
        rec_emb, c_preds = self.forward(X)

        # Perdita per la previsione dei concetti
        concept_loss = self.bce(c_preds, c_true.float())

        # Perdita per la ricostruzione dell'input
        mse_loss = self.mse(rec_emb, X)

        # La perdita totale ora è solo la combinazione tra concetti e ricostruzione
        loss = concept_loss + 0.1 * mse_loss

        # Log delle metriche
        concept_accuracy = roc_auc_score(c_true, c_preds.detach())

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, I, batch_idx):
        X, c_true = self._unpack_input(I)

        # Predizione concetti e ricostruzione
        rec_emb, c_preds = self.forward(X)

        # Perdita per la previsione dei concetti
        concept_loss = self.bce(c_preds, c_true.float())

        # Perdita per la ricostruzione dell'input
        mse_loss = self.mse(rec_emb, X)

        # La perdita totale ora è solo la combinazione tra concetti e ricostruzione
        loss = concept_loss + 0.1 * mse_loss

        # Log delle metriche
        concept_accuracy = roc_auc_score(c_true, c_preds.detach())

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer