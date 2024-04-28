#%%
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm

# Scikit-learn packages
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# Packages to define a RoBERTa model
from transformers import RobertaConfig, AutoTokenizer, TFAutoModel

# Keras and TensorFlow packages
import tensorflow as tf
from keras import backend as K

# Wanddb
import wandb


#%%
def create_model_pipeline(
    df
    ):

    # Importing BERT pre-trained model and tokenizer
    model_name = 'roberta-large'
    config = RobertaConfig.from_pretrained(model_name, output_hidden_states=False)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_name)
    transformer_model = TFAutoModel.from_pretrained(model_name, config = config)

    # function for creating BERT based model
    def create_model(
        nb_labels,
        trf_model
        ):

        # Computing max length of samples
        full_text = df['text_clean']
        max_length = full_text.apply(lambda x: len(x.split())).max()
        print(f"max length of samples: {max_length}")

        # Load the MainLayer
        roberta = trf_model.layers[0]

        # Build the model inputs
        input_ids = tf.keras.layers.Input(shape=(max_length,), name='input_ids', dtype='int32')
        attention_mask = tf.keras.layers.Input(shape=(max_length,), name='attention_mask', dtype='int32')
        token_type_ids = tf.keras.layers.Input(shape=(max_length,), name='token_ids', dtype='int32')
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

        # Load the Transformers BERT model as a layer in a Keras model
        roberta_model = roberta(inputs)[1]
        dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, name='pooled_output')
        pooled_output = dropout(roberta_model, training=False)

        # Then build the model output
        emotion = tf.keras.layers.Dense(units=nb_labels, activation="sigmoid", kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range), name='emotion')(pooled_output)
        outputs = emotion

        # And combine it all in a model object
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='RoBERTa_MultiLabel')

        return model, max_length

    target_cols = [c for c in df.columns if c not in ['text', 'text_clean', 'Unnamed: 0', 'source', 'dataset', 'id']]

    # Creating a model instance
    model, max_length = create_model(len(target_cols), transformer_model)

    return tokenizer, model, max_length, target_cols

#%%
def customTokenizer(
    df: pd.DataFrame,
    tokenizer,
    max_length,
    target_cols,
    batch_size: int = 128
    ):

    # Creating variables
    X = df['text_clean']
    y = df.loc[:, target_cols].values.astype(float)

    # Tokenizing train data
    token = tokenizer(
        text = X.to_list(),
        add_special_tokens = True,
        max_length = max_length,
        truncation = True,
        padding = 'max_length',
        return_tensors = 'tf',
        return_token_type_ids = True,
        return_attention_mask = True,
        verbose = True)

    # Creating BERT compatible inputs with Input Ids, attention masks and token Ids
    model_inputs = {'input_ids': token['input_ids'], 'attention_mask': token['attention_mask'],'token_type_ids': token['token_type_ids']}

    # Creating TF tensors
    tensor = tf.data.Dataset.from_tensor_slices((model_inputs, y)).shuffle(len(model_inputs)).batch(batch_size)

    return X, y, model_inputs, tensor

#%%
def trainModelWandDB(
    modelnm,
    train_tf,
    val_tf,
    ytrain,
    n_epochs: int = 4,
    lr = 5.e-05
    ):

    # Function for calculating multilabel class weights
    def calculating_class_weights(y_true):
        number_dim = np.shape(y_true)[1]
        weights = np.empty([number_dim, 2])
        for i in range(number_dim):
            weights[i] = compute_class_weight('balanced', classes = [0.,1.], y = y_true[:, i])
        return weights

    # Custom loss function for multilabel
    def get_weighted_loss(weights):
        def weighted_loss(y_true, y_pred):
            return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
        return weighted_loss

    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    # Set an optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        epsilon=1e-08
        )

    # Set loss
    class_weights = calculating_class_weights(ytrain)
    loss = get_weighted_loss(class_weights)

    # Compile the model
    modelnm.compile(
        optimizer = optimizer,
        loss = loss,
        metrics=['acc', f1_m])

    # train the model
    history = modelnm.fit(train_tf,
                        epochs=n_epochs,
                        validation_data=val_tf,
                        callbacks=[wandb.keras.WandbMetricsLogger()]
                        )
    
    return modelnm, history

#%%
def evaluate(
    model,
    test,
    y_test, 
    target_cols
    ):
    # from probabilities to labels using a given threshold
    def proba_to_labels(y_pred_proba, threshold=0.85):

        y_pred_labels = np.zeros_like(y_pred_proba)

        for i in range(y_pred_proba.shape[0]):
            for j in range(y_pred_proba.shape[1]):
                if y_pred_proba[i][j] > threshold:
                    y_pred_labels[i][j] = 1
                else:
                    y_pred_labels[i][j] = 0

        return y_pred_labels

    # Model evaluation function
    def model_eval(y_true, y_pred_labels, emotions):

        # Defining variables
        precision = []
        recall = []
        f1 = []

        # Per emotion evaluation
        for i in range(len(emotions)):

            # Computing precision, recall and f1-score
            p, r, f1_score, _ = precision_recall_fscore_support(y_true[:, i], y_pred_labels[:, i], average="binary")

            # Append results in lists
            precision.append(round(p, 2))
            recall.append(round(r, 2))
            f1.append(round(f1_score, 2))

        # Macro evaluation
        macro_p, macro_r, macro_f1_score, _ = precision_recall_fscore_support(y_true, y_pred_labels, average="macro")

        # Append results in lists
        precision.append(round(macro_p, 2))
        recall.append(round(macro_r, 2))
        f1.append(round(macro_f1_score, 2))

        # Converting results to a dataframe
        df_results = pd.DataFrame({"Precision":precision, "Recall":recall, 'F1':f1})
        df_results.index = emotions+['MACRO-AVERAGE']

        return df_results
    
    # Function that computes labels from probabilities and optimizes the threshold that maximizes f1-score
    def proba_to_labels_opt(y_true, y_pred_proba):

        '''
        Inputs:
            y_true: Ground truth labels
            y_pred_proba: predicted probabilities

        Outputs :
            best_y_pred_labels: preticted labels associated with best threshold
            best_t: best threshold
            best_macro_f1: macro f1-score associated with predicted labels
        '''

        # range of possible thresholds
        thresholds = np.arange(0.50, 0.99, 0.01)

        # Computing threshold that maximizes macro f1-score
        best_y_pred_labels = np.zeros_like(y_pred_proba)
        best_t = 0
        best_macro_f1 = 0

        # Iterating through possible thresholds
        for t in thresholds:

            y_pred_labels = proba_to_labels(y_pred_proba, t)

            _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred_labels, average="macro")

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_t = t
                best_y_pred_labels = y_pred_labels

        return best_y_pred_labels, best_t, best_macro_f1

    # Making probability predictions on test data
    y_pred_proba = model.predict(test)

    # Generate labels
    y_pred_labels_opt, threshold_opt, macro_f1_opt = proba_to_labels_opt(y_test, y_pred_proba)
    print("The model's threshold is {}".format(threshold_opt))
    print("The model's best macro-f1 is {}".format(macro_f1_opt))

    # Model evaluation
    eval_df = model_eval(y_test, y_pred_labels_opt, target_cols)

    return eval_df

#%%
class GenRSModelWandDB():
    def __init__(self, name, wanddb_config, df_reference, df_target=None, n_epochs=4, lr = 5.e-05, batch_size = 128):
        self.df_reference, self.name, self.n_epochs, self.lr, self.batch_size = df_reference.copy(), name, n_epochs, lr, batch_size

        df = df_reference.copy()

        if isinstance(df_target, pd.DataFrame):
            self.df_test = df_target[df_target['dataset']=='test'].copy()
            self.df_val = df_target[df_target['dataset']=='val'].copy()
        else:
            self.df_test = df[df['dataset']=='test']
            self.df_val = df[df['dataset']=='val']

        print(f"MODEL NAME: {(self.name).upper()}")

        project = wanddb_config.pop('project')

        wandb.init(
            # set the wandb project where this run will be logged
            project=project,
            # track hyperparameters and run metadata
            config={**{
            "learning_rate": lr,
            "model": "RoBERTa-large",
            "epochs": n_epochs,
            "batch size": batch_size
            }, **wanddb_config}
        )

    def createModel(self):
        df = (self.df_reference).copy()

        # Create model
        print('*** CREATING MODEL ***')
        self.tokenizer, self.model, self.max_length, self.target_cols = create_model_pipeline(df)

        # Tokenise
        print('*** TOKENISING DATA ***')
        X_train, y_train, train, train_tensor = customTokenizer(df[df['dataset']=='train'], self.tokenizer, self.max_length, self.target_cols, self.batch_size)
        X_val, y_val, val, val_tensor = customTokenizer(self.df_val, self.tokenizer, self.max_length, self.target_cols, self.batch_size)
        X_test, y_test, test, test_tensor = customTokenizer(self.df_test, self.tokenizer, self.max_length, self.target_cols, self.batch_size)

        # Train
        print('*** TRAINING MODEL ***')
        self.model, self.history = trainModelWandDB(self.model, train_tensor, val_tensor, y_train, self.n_epochs, self.lr)

        # Evaluate
        print('*** EVALUATING MODEL ***')
        self.eval_reference = evaluate(self.model, test, y_test, self.target_cols)