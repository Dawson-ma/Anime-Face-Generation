# use resource in https://github.com/lucidrains/stylegan2-pytorch
from stylegan2_pytorch.cli import train_from_folder
from stylegan2_pytorch.stylegan2_pytorch import Trainer, NanException
from tqdm import tqdm

# Hyoerparameters
data = './faces'
results_dir = './results'
models_dir = './models'
name = 'stylegan2'
new = False
load_from = -1
image_size = 64
network_capacity = 16
fmap_max = 512
transparent = False
batch_size = 5
gradient_accumulate_every = 6
num_train_steps = 200000
learning_rate = 2e-4
lr_mlp = 0.1
ttur_mult = 1.5
rel_disc_loss = False
num_workers =  None
save_every = 1000
evaluate_every = 1000
generate = False
num_generate = 1000
generate_interpolation = False
interpolation_num_steps = 100
save_frames = False
num_image_tiles = 8
trunc_psi = 0.75
mixed_prob = 0.9
fp16 = False
no_pl_reg = False
cl_reg = False
fq_layers = []
fq_dict_size = 256
attn_layers = []
no_const = False
aug_prob = 0.
top_k_training = False
generator_top_k_gamma = 0.99
generator_top_k_frac = 0.5
dual_contrast_loss = False
dataset_aug_prob = 0.
multi_gpus = False
calculate_fid_every = None
calculate_fid_num_images = 12800
clear_fid_cache = False
seed = 42
log = False

if __name__== '__main__':
    train_from_folder(
        data = data,
        results_dir = results_dir,
        models_dir = models_dir,
        name = name,
        new = new,
        load_from = load_from,
        image_size = image_size,
        network_capacity = network_capacity,
        fmap_max = fmap_max,
        transparent = transparent,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        num_train_steps = num_train_steps,
        learning_rate = learning_rate,
        lr_mlp = lr_mlp,
        ttur_mult = ttur_mult,
        rel_disc_loss = rel_disc_loss,
        num_workers =  num_workers,
        save_every = save_every,
        evaluate_every = evaluate_every,
        generate = generate,
        num_generate = num_generate,
        generate_interpolation = generate_interpolation,
        interpolation_num_steps = interpolation_num_steps,
        save_frames = save_frames,
        num_image_tiles = num_image_tiles,
        trunc_psi = trunc_psi,
        mixed_prob = mixed_prob,
        fp16 = fp16,
        no_pl_reg = no_pl_reg,
        cl_reg = cl_reg,
        fq_layers = fq_layers,
        fq_dict_size = fq_dict_size,
        attn_layers = attn_layers,
        no_const = no_const,
        aug_prob = aug_prob,
        top_k_training = top_k_training,
        generator_top_k_gamma = generator_top_k_gamma,
        generator_top_k_frac = generator_top_k_frac,
        dual_contrast_loss = dual_contrast_loss,
        dataset_aug_prob = dataset_aug_prob,
        multi_gpus = multi_gpus,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache,
        seed = seed,
        log = log
    )

    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        network_capacity = network_capacity,
        fmap_max = fmap_max,
        transparent = transparent,
        lr = learning_rate,
        lr_mlp = lr_mlp,
        ttur_mult = ttur_mult,
        rel_disc_loss = rel_disc_loss,
        num_workers = num_workers,
        save_every = save_every,
        evaluate_every = evaluate_every,
        num_image_tiles = num_image_tiles,
        trunc_psi = trunc_psi,
        fp16 = fp16,
        no_pl_reg = no_pl_reg,
        cl_reg = cl_reg,
        fq_layers = fq_layers,
        fq_dict_size = fq_dict_size,
        attn_layers = attn_layers,
        no_const = no_const,
        aug_prob = aug_prob,
        top_k_training = top_k_training,
        generator_top_k_gamma = generator_top_k_gamma,
        generator_top_k_frac = generator_top_k_frac,
        dual_contrast_loss = dual_contrast_loss,
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache,
        mixed_prob = mixed_prob,
        log = log
    )

    num_image_tiles = 1
    model = Trainer(**model_args)
    model.load(load_from)
    for num in tqdm(range(num_generate)):
        model.evaluate(f'{num}', num_image_tiles)