use std::{fmt::Debug, rc::Rc};

// TODO:
//  - Check whether all the `cases` have unique linear equations.
//  - Check whether any equations are exclusive to each other.
//  - Check whether coefficients of all input variables, cover complete Real number
//    space, in all `cases`.
//  - Take arguments from CLI.

/// This project is for calculating how many unique linear equations can be used by a
/// Multi-layer Perceptron model simultaneously.
/// Model architecture can be perceived as following. ReLU activation function is assumed
/// after every hidden layer, i.e. first layer to second last layer. ReLU activations
/// of second last layer are converged into a single neuron in the last, AKA "output",
/// layer so that model outputs a single number.
fn main() {
    let num_inputs: Uint = 1;
    let perceptrons_per_layer: &[Uint] = &[1, 1, 1];
    let should_print_equations = true;

    let mut mlp = MLP::new(num_inputs, perceptrons_per_layer);
    mlp.calculate_cases(should_print_equations);
    let counted_constant_cases = mlp.count_constant_cases();
    let derived_constant_cases = formula_constant_cases(perceptrons_per_layer);
    println!(
        "Is derived_constant_cases same: {} , {}",
        counted_constant_cases == derived_constant_cases,
        derived_constant_cases
    );
}

fn formula_constant_cases(perceptrons_per_layer: &[Uint]) -> usize {
    let mut a: usize = 0;
    for c in 0..perceptrons_per_layer.len() {
        let mut b: usize = 1;
        for j in 0..(c) {
            b *= usize::pow(2, perceptrons_per_layer[j].into()) - 1;
        }
        b *= usize::pow(
            2,
            perceptrons_per_layer[(c + 1)..perceptrons_per_layer.len()]
                .iter()
                .sum::<Uint>() as u32,
        ) as usize;

        a += b;
    }

    return a;
}

type Uint = u16;

#[derive(Debug)]
struct MLP {
    layers: Vec<Layer>,
    cases: Vec<Case>,
    total_cases_num: usize,
    total_hidden_perceptrons: Uint,
    num_inputs: Uint,
}

impl MLP {
    fn new(num_inputs: Uint, perceptrons_per_layer: &[Uint]) -> Self {
        let mut layers = Vec::<Layer>::with_capacity(perceptrons_per_layer.len());
        let first_layer = Layer::new(0, num_inputs, 0);
        layers.push(first_layer);
        for (idx, &num_perceptrons) in perceptrons_per_layer.iter().enumerate() {
            let next_layer = Layer::new(
                layers[idx].perceptrons.len() as Uint,
                num_perceptrons,
                idx as Uint + 1,
            );
            layers.push(next_layer);
        }
        let last_layer = Layer::new(
            *perceptrons_per_layer.last().unwrap(),
            1,
            perceptrons_per_layer.len() as Uint + 1,
        );
        layers.push(last_layer);

        let total_hidden_perceptrons: Uint = perceptrons_per_layer.iter().sum();
        let total_cases_num = usize::pow(2, total_hidden_perceptrons.into());
        let cases = Vec::<Case>::with_capacity(total_cases_num.into());

        return MLP {
            layers,
            cases,
            total_cases_num,
            total_hidden_perceptrons,
            num_inputs,
        };
    }

    fn calculate_cases(&mut self, should_print_equations: bool) {
        let discard_bits = usize::BITS - self.total_hidden_perceptrons as u32;
        if discard_bits <= 0 {
            panic!("Reduce the number of perceptrons.")
        }

        for case_num in 0..self.total_cases_num {
            let mut relu_flags =
                Vec::<bool>::with_capacity(self.total_hidden_perceptrons as usize + 1);
            let mut layer_params =
                Vec::<Option<Coefficient>>::with_capacity(self.num_inputs.into());
            for perceptron in self.layers[0].perceptrons.iter() {
                let input_param =
                    Coefficient(vec![WeightProduct(vec![Rc::clone(&perceptron.weights[0])])]);
                layer_params.push(Some(input_param));
            }
            let mut bit_flags = (case_num as usize) << discard_bits;
            // Add bit flag for output layer perceptron to True.
            bit_flags = bit_flags | usize::pow(2, discard_bits - 1);
            for layer in self.layers.iter().skip(1) {
                let mut layer_output =
                    Vec::<Option<Coefficient>>::with_capacity(layer.perceptrons.len());
                for perceptron in layer.perceptrons.iter() {
                    let mut temp_layer_params = layer_params.clone();
                    let flag = (bit_flags & usize::pow(2, usize::BITS - 1)) != 0;
                    bit_flags <<= 1;
                    relu_flags.push(flag);

                    if flag {
                        let mut output = Coefficient(vec![]);
                        // Add Bias.
                        output.add(Coefficient(vec![WeightProduct(vec![Rc::clone(
                            &perceptron.weights[0],
                        )])]));

                        for weight in perceptron.weights.iter().skip(1).rev() {
                            let param = temp_layer_params.pop().unwrap();
                            match param {
                                Some(mut param) => {
                                    param.multiply(weight);
                                    output.add(param);
                                }
                                None => {}
                            }
                        }
                        layer_output.push(Some(output));
                    } else {
                        layer_output.push(None);
                    }
                }
                layer_params = layer_output;
            }

            let mut coefficients = Vec::<Coefficient>::with_capacity((self.num_inputs + 1).into());
            for _ in 0..self.num_inputs + 1 {
                coefficients.push(Coefficient(vec![]));
            }
            for weight_product in layer_params[0].take().unwrap().0 {
                'weights_loop: for (weight_i, weight) in weight_product.0.iter().enumerate() {
                    for perceptron in self.layers[0].perceptrons.iter() {
                        if Rc::ptr_eq(&perceptron.weights[0], weight) {
                            let mut weight_product = weight_product.clone();
                            weight_product.0.remove(weight_i);

                            coefficients[perceptron.idx.idx as usize]
                                .add(Coefficient(vec![weight_product]));
                            break 'weights_loop;
                        }
                    }
                    // If the weight_product doesn't belong to any input param, then it's part of bias.
                    if weight_i == weight_product.0.len() - 1 {
                        coefficients[0].add(Coefficient(vec![weight_product.clone()]));
                    }
                }
            }

            // Last ReLU flag is representing output layer, hence will always be True.
            relu_flags.pop();

            self.cases.push(Case {
                coefficients,
                relu_flags,
            })
        }

        if should_print_equations {
            self.print_equations();
        }
    }

    fn print_equations(&self) {
        println!("Start {{");
        for case in self.cases.iter().rev() {
            for (flag_i, flag) in case.relu_flags.iter().enumerate() {
                print!("{}", usize::from(*flag));
                let mut flag_i = flag_i as isize + 1;
                for layer in self.layers.iter().skip(1) {
                    flag_i -= layer.perceptrons.len() as isize;
                    if flag_i < 0 {
                        break;
                    } else if flag_i == 0 {
                        print!("  ");
                        break;
                    }
                }
            }
            print!("\x08:\n"); // `\x08` is backspace.

            for (mut input_i, coefficient) in case.coefficients.iter().skip(1).enumerate() {
                input_i += 1; // Skipped 1 element.
                print!("(");
                for weight_product in coefficient.0.iter().rev() {
                    for weight in weight_product.0.iter().rev() {
                        print!(
                            // escape codes are for making dot product symbol bold.
                            "w_{}{}{}\x1b[1m⋅\x1b[0m",
                            weight.idx.perceptron_idx.layer_idx,
                            weight.idx.perceptron_idx.idx,
                            weight.idx.idx
                        );
                    }
                    print!("\x08 + ");
                }
                if !coefficient.0.is_empty() {
                    print!("\x08\x08\x08");
                }
                print!(")\x1b[1m⋅\x1b[0mx_{} \x1b[1m+\x1b[0m\n", input_i);
            }
            print!("(");
            for bias_product in case.coefficients[0].0.iter().rev() {
                for weight in bias_product.0.iter().rev() {
                    if weight.idx.idx != 0 {
                        print!(
                            "w_{}{}{}\x1b[1m⋅\x1b[0m",
                            weight.idx.perceptron_idx.layer_idx,
                            weight.idx.perceptron_idx.idx,
                            weight.idx.idx
                        );
                    } else {
                        print!(
                            "b_{}{}\x1b[1m⋅\x1b[0m",
                            weight.idx.perceptron_idx.layer_idx, weight.idx.perceptron_idx.idx
                        );
                    }
                }
                print!("\x08 + ");
            }
            print!("\x08\x08\x08) \n\n");
        }
        println!("}} End");
    }

    fn count_constant_cases(&self) -> usize {
        // let mut constant_cases = 0;
        let mut theory_constant_cases = 0;
        for case in self.cases.iter() {
            /* let mut num_of_input_param_coefficients = 0;
            for coefficient in case.coefficients.iter().skip(1) {
                num_of_input_param_coefficients += coefficient.0.len();
            }
            if num_of_input_param_coefficients == 0 {
                constant_cases += 1;
            } */
            if case.coefficients[1].0.len() == 0 {
                theory_constant_cases += 1;
            }
        }

        // assert_eq!(constant_cases, theory_constant_cases);
        println!("Number of constant cases are: {}", theory_constant_cases);

        return theory_constant_cases;
    }
}

#[derive(Debug)]
struct Case {
    coefficients: Vec<Coefficient>,
    relu_flags: Vec<bool>,
}

#[derive(Debug, Clone)]
struct WeightProduct(Vec<Rc<Weight>>);

impl WeightProduct {
    fn multiply(&mut self, weight: &Rc<Weight>) {
        self.0.push(Rc::clone(weight));
    }
}

/// Sum of `WeightProduct` terms.
#[derive(Debug, Clone)]
struct Coefficient(Vec<WeightProduct>);

impl Coefficient {
    fn add(&mut self, coefficient: Coefficient) {
        for product in coefficient.0 {
            self.0.push(product);
        }
    }

    fn multiply(&mut self, weight: &Rc<Weight>) {
        for product in self.0.iter_mut() {
            product.multiply(weight);
        }
    }
}

#[derive(Debug)]
struct Layer {
    perceptrons: Vec<Perceptron>,
}

impl Layer {
    fn new(num_inputs: Uint, num_perceptrons: Uint, layer_idx: Uint) -> Self {
        let mut perceptrons = Vec::<Perceptron>::with_capacity(num_perceptrons.into());
        for idx in 0..num_perceptrons {
            let perceptron_idx = PerceptronIdx {
                idx: idx + 1,
                layer_idx,
            };
            let next_perceptron = Perceptron::new(num_inputs, perceptron_idx);
            perceptrons.push(next_perceptron);
        }

        return Layer { perceptrons };
    }
}

#[derive(Debug)]
struct Perceptron {
    idx: PerceptronIdx,
    weights: Vec<Rc<Weight>>,
}

impl Perceptron {
    fn new(num_inputs: Uint, perceptron_idx: PerceptronIdx) -> Self {
        let mut weights = Vec::<Rc<Weight>>::with_capacity(num_inputs as usize + 1);
        // Bias is the first weight.
        let bias_idx = WeightIdx {
            idx: 0,
            perceptron_idx,
        };
        let bias = Weight::new(bias_idx);
        weights.push(Rc::new(bias));

        for idx in 0..num_inputs {
            let weight_idx = WeightIdx {
                idx: idx + 1,
                perceptron_idx,
            };
            let next_weight = Weight::new(weight_idx);
            weights.push(Rc::new(next_weight));
        }

        return Perceptron {
            idx: perceptron_idx,
            weights,
        };
    }
}

#[derive(Debug)]
struct Weight {
    idx: WeightIdx,
}

impl Weight {
    fn new(weight_idx: WeightIdx) -> Self {
        return Weight { idx: weight_idx };
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct PerceptronIdx {
    idx: Uint,
    layer_idx: Uint,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct WeightIdx {
    idx: Uint,
    perceptron_idx: PerceptronIdx,
}
