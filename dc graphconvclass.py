class GraphConvModel(KerasModel):
    """Graph Convolutional Models.

    This class implements the graph convolutional model from the
    following paper [1]_. These graph convolutions start with a per-atom set of
    descriptors for each atom in a molecule, then combine and recombine these
    descriptors over convolutional layers.
    following [1]_.


    References
    ----------
    .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
        learning molecular fingerprints." Advances in neural information processing
        systems. 2015.
    """

    def __init__(self,
                 n_tasks: int,
                 graph_conv_layers: List[int] = [64, 64],
                 dense_layer_size: int = 128,
                 dropout: float = 0.0,
                 mode: str = "classification",
                 number_atom_features: int = 75,
                 n_classes: int = 2,
                 batch_size: int = 100,
                 batch_normalize: bool = True,
                 uncertainty: bool = False,
                 **kwargs):
        """The wrapper class for graph convolutions.

        Note that since the underlying _GraphConvKerasModel class is
        specified using imperative subclassing style, this model
        cannout make predictions for arbitrary outputs.

        Parameters
        ----------
        n_tasks: int
            Number of tasks
        graph_conv_layers: list of int
            Width of channels for the Graph Convolution Layers
        dense_layer_size: int
            Width of channels for Atom Level Dense Layer after GraphPool
        dropout: list or float
            the dropout probablity to use for each layer.  The length of this list
            should equal len(graph_conv_layers)+1 (one value for each convolution
            layer, and one for the dense layer).  Alternatively this may be a single
            value instead of a list, in which case the same value is used for every
            layer.
        mode: str
            Either "classification" or "regression"
        number_atom_features: int
            75 is the default number of atom features created, but
            this can vary if various options are passed to the
            function atom_features in graph_features
        n_classes: int
            the number of classes to predict (only used in classification mode)
        batch_normalize: True
            if True, apply batch normalization to model
        uncertainty: bool
            if True, include extra outputs and loss terms to enable the uncertainty
            in outputs to be predicted
        """
        self.mode = mode
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.uncertainty = uncertainty
        model = _GraphConvKerasModel(n_tasks,
                                     graph_conv_layers=graph_conv_layers,
                                     dense_layer_size=dense_layer_size,
                                     dropout=dropout,
                                     mode=mode,
                                     number_atom_features=number_atom_features,
                                     n_classes=n_classes,
                                     batch_normalize=batch_normalize,
                                     uncertainty=uncertainty,
                                     batch_size=batch_size)
        if mode == "classification":
            output_types = ['prediction', 'loss', 'embedding']
            loss: Union[Loss, LossFn] = SoftmaxCrossEntropy()
        else:
            if self.uncertainty:
                output_types = [
                    'prediction', 'variance', 'loss', 'loss', 'embedding'
                ]

                def loss(outputs, labels, weights):
                    output, labels = dc.models.losses._make_tf_shapes_consistent(
                        outputs[0], labels[0])
                    output, labels = dc.models.losses._ensure_float(
                        output, labels)
                    losses = tf.square(output - labels) / tf.exp(
                        outputs[1]) + outputs[1]
                    w = weights[0]
                    if len(w.shape) < len(losses.shape):
                        if tf.is_tensor(w):
                            shape = tuple(w.shape.as_list())
                        else:
                            shape = w.shape
                        shape = tuple(-1 if x is None else x for x in shape)
                        w = tf.reshape(
                            w,
                            shape + (1,) * (len(losses.shape) - len(w.shape)))
                    return tf.reduce_mean(losses * w) + sum(self.model.losses)
            else:
                output_types = ['prediction', 'embedding']
                loss = L2Loss()
        super(GraphConvModel, self).__init__(model,
                                             loss,
                                             output_types=output_types,
                                             batch_size=batch_size,
                                             **kwargs)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                if y_b is not None and self.mode == 'classification' and not (
                        mode == 'predict'):
                    y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                        -1, self.n_tasks, self.n_classes)
                multiConvMol = ConvMol.agglomerate_mols(X_b)
                n_samples = np.array(X_b.shape[0])
                inputs = [
                    multiConvMol.get_atom_features(), multiConvMol.deg_slice,
                    np.array(multiConvMol.membership), n_samples
                ]
                for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                    inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
                yield (inputs, [y_b], [w_b])