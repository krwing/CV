import gc
from operator import getitem
import cv2
import numpy as np
import skimage.measure
from PIL import Image
import torch
from torchvision.transforms import Compose, transforms
from transforms_resize import Resize, NormalizeImage, PrepareForNet
from src import backbone

device = torch.device("cuda:0")

def estimateleres(img, model, w, h):
    # leres transform input
    rgb_c = img[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (w, h))
    img_torch = scale_torch(A_resize)[None, :, :, :]

    # compute
    with torch.no_grad():
        if device == torch.device("cuda"):
            img_torch = img_torch.cuda()
        prediction = model.depth_model(img_torch)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return prediction


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        img = transform(img.astype(np.float32))
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


def estimatezoedepth(img, model, w, h):
    # x = transforms.ToTensor()(img).unsqueeze(0)
    # x = x.type(torch.float32)
    # x.to(device)
    # prediction = model.infer(x)
    model.core.prep.resizer._Resize__width = w
    model.core.prep.resizer._Resize__height = h
    prediction = model.infer_pil(img)

    return prediction


def estimatemidas(img, model, w, h, resize_mode, normalization, no_half, precision_is_autocast):
    import contextlib
    # init transform
    transform = Compose(
        [
            Resize(
                w,
                h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    # transform input
    img_input = transform({"image": img})["image"]

    # compute
    precision_scope = torch.autocast if precision_is_autocast and device == torch.device(
        "cuda") else contextlib.nullcontext
    with torch.no_grad(), precision_scope("cuda"):
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            if not no_half:
                sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def estimatemarigold(image, model, w, h, marigold_ensembles=5, marigold_steps=12):
    # This hideous thing should be re-implemented once there is support from the upstream.
    img = cv2.cvtColor((image * 255.0001).astype('uint8'), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    with torch.no_grad():
        pipe_out = model(img, processing_res=w, show_progress_bar=False,
                         ensemble_size=marigold_ensembles, denoising_steps=marigold_steps,
                         match_input_res=False)
        return cv2.resize(pipe_out.depth_np, (image.shape[:2][::-1]), interpolation=cv2.INTER_CUBIC)


def estimatedepthanything(image, model, w, h):
    from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    transform = Compose(
        [
            Resize(
                width=w // 14 * 14,
                height=h // 14 * 14,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    timage = transform({"image": image})["image"]
    timage = torch.from_numpy(timage).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        depth = model(timage)
    import torch.nn.functional as F
    depth = F.interpolate(
        depth[None], (image.shape[0], image.shape[1]), mode="bilinear", align_corners=False
    )[0, 0]

    return depth.cpu().numpy()


class ImageandPatchs:
    def __init__(self, root_dir, name, patchsinfo, rgb_image, scale=1):
        self.root_dir = root_dir
        self.patchsinfo = patchsinfo
        self.name = name
        self.patchs = patchsinfo
        self.scale = scale

        self.rgb_image = cv2.resize(rgb_image, (round(rgb_image.shape[1] * scale), round(rgb_image.shape[0] * scale)),
                                    interpolation=cv2.INTER_CUBIC)

        self.do_have_estimate = False
        self.estimation_updated_image = None
        self.estimation_base_image = None

    def __len__(self):
        return len(self.patchs)

    def set_base_estimate(self, est):
        self.estimation_base_image = est
        if self.estimation_updated_image is not None:
            self.do_have_estimate = True

    def set_updated_estimate(self, est):
        self.estimation_updated_image = est
        if self.estimation_base_image is not None:
            self.do_have_estimate = True

    def __getitem__(self, index):
        patch_id = int(self.patchs[index][0])
        rect = np.array(self.patchs[index][1]['rect'])
        msize = self.patchs[index][1]['size']

        ## applying scale to rect:
        rect = np.round(rect * self.scale)
        rect = rect.astype('int')
        msize = round(msize * self.scale)

        patch_rgb = impatch(self.rgb_image, rect)
        if self.do_have_estimate:
            patch_whole_estimate_base = impatch(self.estimation_base_image, rect)
            patch_whole_estimate_updated = impatch(self.estimation_updated_image, rect)
            return {'patch_rgb': patch_rgb, 'patch_whole_estimate_base': patch_whole_estimate_base,
                    'patch_whole_estimate_updated': patch_whole_estimate_updated, 'rect': rect,
                    'size': msize, 'id': patch_id}
        else:
            return {'patch_rgb': patch_rgb, 'rect': rect, 'size': msize, 'id': patch_id}

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #    torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


def impatch(image, rect):
    # Extract the given patch pixels from a given image.
    w1 = rect[0]
    h1 = rect[1]
    w2 = w1 + rect[2]
    h2 = h1 + rect[3]
    image_patch = image[h1:h2, w1:w2]
    return image_patch


class ImageandPatchs:
    def __init__(self, root_dir, name, patchsinfo, rgb_image, scale=1):
        self.root_dir = root_dir
        self.patchsinfo = patchsinfo
        self.name = name
        self.patchs = patchsinfo
        self.scale = scale

        self.rgb_image = cv2.resize(rgb_image, (round(rgb_image.shape[1] * scale), round(rgb_image.shape[0] * scale)),
                                    interpolation=cv2.INTER_CUBIC)

        self.do_have_estimate = False
        self.estimation_updated_image = None
        self.estimation_base_image = None

    def __len__(self):
        return len(self.patchs)

    def set_base_estimate(self, est):
        self.estimation_base_image = est
        if self.estimation_updated_image is not None:
            self.do_have_estimate = True

    def set_updated_estimate(self, est):
        self.estimation_updated_image = est
        if self.estimation_base_image is not None:
            self.do_have_estimate = True

    def __getitem__(self, index):
        patch_id = int(self.patchs[index][0])
        rect = np.array(self.patchs[index][1]['rect'])
        msize = self.patchs[index][1]['size']

        ## applying scale to rect:
        rect = np.round(rect * self.scale)
        rect = rect.astype('int')
        msize = round(msize * self.scale)

        patch_rgb = impatch(self.rgb_image, rect)
        if self.do_have_estimate:
            patch_whole_estimate_base = impatch(self.estimation_base_image, rect)
            patch_whole_estimate_updated = impatch(self.estimation_updated_image, rect)
            return {'patch_rgb': patch_rgb, 'patch_whole_estimate_base': patch_whole_estimate_base,
                    'patch_whole_estimate_updated': patch_whole_estimate_updated, 'rect': rect,
                    'size': msize, 'id': patch_id}
        else:
            return {'patch_rgb': patch_rgb, 'rect': rect, 'size': msize, 'id': patch_id}

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'



    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #    torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


def estimateboost(img, model, model_type, pix2pixmodel, whole_size_threshold):
    pix2pixsize = 1024
    net_receptive_field_size = 384
    patch_netsize = 2 * net_receptive_field_size
    gc.collect()
    backbone.torch_gc()
    mask_org = generatemask((3000, 3000))
    mask = mask_org.copy()
    r_threshold_value = 0.2
    input_resolution = img.shape
    scale_threshold = 3
    whole_image_optimal_size, patch_scale = calculateprocessingres(img, net_receptive_field_size, r_threshold_value,
                                                                   scale_threshold, whole_size_threshold)

    whole_estimate = doubleestimate(img, net_receptive_field_size, whole_image_optimal_size, pix2pixsize, model,
                                    model_type, pix2pixmodel)

    factor = max(min(1, 4 * patch_scale * whole_image_optimal_size / whole_size_threshold), 0.2)
    if img.shape[0] > img.shape[1]:
        a = 2 * whole_image_optimal_size
        b = round(2 * whole_image_optimal_size * img.shape[1] / img.shape[0])
    else:
        a = round(2 * whole_image_optimal_size * img.shape[0] / img.shape[1])
        b = 2 * whole_image_optimal_size
    b = int(round(b / factor))
    a = int(round(a / factor))
    img = cv2.resize(img, (b, a), interpolation=cv2.INTER_CUBIC)
    base_size = net_receptive_field_size * 2
    patchset = generatepatchs(img, base_size, factor)

    mergein_scale = input_resolution[0] / img.shape[0]

    imageandpatchs = ImageandPatchs('', '', patchset, img, mergein_scale)
    whole_estimate_resized = cv2.resize(whole_estimate, (round(img.shape[1] * mergein_scale),
                                                         round(img.shape[0] * mergein_scale)),
                                        interpolation=cv2.INTER_CUBIC)
    imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
    imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())


    for patch_ind in range(len(imageandpatchs)):

        # Get patch information
        patch = imageandpatchs[patch_ind]  # patch object
        patch_rgb = patch['patch_rgb']  # rgb patch
        patch_whole_estimate_base = patch['patch_whole_estimate_base']  # corresponding patch from base
        rect = patch['rect']  # patch size and location
        patch_id = patch['id']  # patch ID
        org_size = patch_whole_estimate_base.shape  # the original size from the unscaled input
        print('\t processing patch', patch_ind, '/', len(imageandpatchs) - 1, '|', rect)

        patch_estimation = doubleestimate(patch_rgb, net_receptive_field_size, patch_netsize, pix2pixsize, model,
                                          model_type, pix2pixmodel)
        patch_estimation = cv2.resize(patch_estimation, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)
        patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (pix2pixsize, pix2pixsize),
                                               interpolation=cv2.INTER_CUBIC)

        pix2pixmodel.set_input(patch_whole_estimate_base, patch_estimation)

        # Run merging network
        pix2pixmodel.test()
        visuals = pix2pixmodel.get_current_visuals()

        prediction_mapped = visuals['fake_B']
        prediction_mapped = (prediction_mapped + 1) / 2
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

        mapped = prediction_mapped

        p_coef = np.polyfit(mapped.reshape(-1), patch_whole_estimate_base.reshape(-1), deg=1)
        merged = np.polyval(p_coef, mapped.reshape(-1)).reshape(mapped.shape)

        merged = cv2.resize(merged, (org_size[1], org_size[0]), interpolation=cv2.INTER_CUBIC)

        # Get patch size and location
        w1 = rect[0]
        h1 = rect[1]
        w2 = w1 + rect[2]
        h2 = h1 + rect[3]

        if mask.shape != org_size:
            mask = cv2.resize(mask_org, (org_size[1], org_size[0]), interpolation=cv2.INTER_LINEAR)

        tobemergedto = imageandpatchs.estimation_updated_image

        tobemergedto[h1:h2, w1:w2] = np.multiply(tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
        imageandpatchs.set_updated_estimate(tobemergedto)

    return cv2.resize(imageandpatchs.estimation_updated_image, (input_resolution[1], input_resolution[0]),
                      interpolation=cv2.INTER_CUBIC)


def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0] / 16)
    k_size = int(2 * np.ceil(2 * int(size[0] / 16)) + 1)
    mask[int(0.15 * size[0]):size[0] - int(0.15 * size[0]), int(0.15 * size[1]): size[1] - int(0.15 * size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask


def rgb2gray(rgb):
    # Converts rgb to gray
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def resizewithpool(img, size):
    i_size = img.shape[0]
    n = int(np.floor(i_size / size))

    out = skimage.measure.block_reduce(img, (n, n), np.max)
    return out


def calculateprocessingres(img, basesize, confidence=0.1, scale_threshold=3, whole_size_threshold=3000):
    speed_scale = 32
    image_dim = int(min(img.shape[0:2]))

    gray = rgb2gray(img)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad = cv2.resize(grad, (image_dim, image_dim), cv2.INTER_AREA)

    m = grad.min()
    M = grad.max()
    middle = m + (0.4 * (M - m))
    grad[grad < middle] = 0
    grad[grad >= middle] = 1

    kernel = np.ones((int(basesize / speed_scale), int(basesize / speed_scale)), float)
    kernel2 = np.ones((int(basesize / (4 * speed_scale)), int(basesize / (4 * speed_scale))), float)

    threshold = min(whole_size_threshold, scale_threshold * max(img.shape[:2]))

    outputsize_scale = basesize / speed_scale
    for p_size in range(int(basesize / speed_scale), int(threshold / speed_scale), int(basesize / (2 * speed_scale))):
        grad_resized = resizewithpool(grad, p_size)
        grad_resized = cv2.resize(grad_resized, (p_size, p_size), cv2.INTER_NEAREST)
        grad_resized[grad_resized >= 0.5] = 1
        grad_resized[grad_resized < 0.5] = 0

        dilated = cv2.dilate(grad_resized, kernel, iterations=1)
        meanvalue = (1 - dilated).mean()
        if meanvalue > confidence:
            break
        else:
            outputsize_scale = p_size

    grad_region = cv2.dilate(grad_resized, kernel2, iterations=1)
    patch_scale = grad_region.mean()

    return int(outputsize_scale * speed_scale), patch_scale


def doubleestimate(img, size1, size2, pix2pixsize, model, net_type, pix2pixmodel):
    estimate1 = singleestimate(img, size1, model, net_type)
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    estimate2 = singleestimate(img, size2, model, net_type)
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped + 1) / 2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
            torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


def singleestimate(img, msize, model, net_type):
    if net_type == 0:
        return estimateleres(img, model, msize, msize)
    elif net_type == 10:
        return estimatemarigold(img, model, msize, msize)
    elif net_type == 11:
        return estimatedepthanything(img, model, msize, msize)
    elif net_type >= 7:
        # np to PIL
        return estimatezoedepth(Image.fromarray(np.uint8(img * 255)).convert('RGB'), model, msize, msize)
    else:
        return estimatemidasBoost(img, model, msize, msize)


def generatepatchs(img, base_size, factor):
    img_gray = rgb2gray(img)
    whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) + \
                 np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))

    threshold = whole_grad[whole_grad > 0].mean()
    whole_grad[whole_grad < threshold] = 0

    gf = whole_grad.sum() / len(whole_grad.reshape(-1))
    grad_integral_image = cv2.integral(whole_grad)

    blsize = int(round(base_size / 2))
    stride = int(round(blsize * 0.75))

    patch_bound_list = applyGridpatch(blsize, stride, img, [0, 0, 0, 0])

    patch_bound_list = adaptiveselection(grad_integral_image, patch_bound_list, gf, factor)

    patchset = sorted(patch_bound_list.items(), key=lambda x: getitem(x[1], 'size'), reverse=True)
    return patchset


def applyGridpatch(blsize, stride, img, box):
    counter1 = 0
    patch_bound_list = {}
    for k in range(blsize, img.shape[1] - blsize, stride):
        for j in range(blsize, img.shape[0] - blsize, stride):
            patch_bound_list[str(counter1)] = {}
            patchbounds = [j - blsize, k - blsize, j - blsize + 2 * blsize, k - blsize + 2 * blsize]
            patch_bound = [box[0] + patchbounds[1], box[1] + patchbounds[0], patchbounds[3] - patchbounds[1],
                           patchbounds[2] - patchbounds[0]]
            patch_bound_list[str(counter1)]['rect'] = patch_bound
            patch_bound_list[str(counter1)]['size'] = patch_bound[2]
            counter1 = counter1 + 1
    return patch_bound_list

def adaptiveselection(integral_grad, patch_bound_list, gf, factor):
    patchlist = {}
    count = 0
    height, width = integral_grad.shape

    search_step = int(32 / factor)

    for c in range(len(patch_bound_list)):
        # Get patch
        bbox = patch_bound_list[str(c)]['rect']

        cgf = getGF_fromintegral(integral_grad, bbox) / (bbox[2] * bbox[3])

        if cgf >= gf:
            bbox_test = bbox.copy()
            patchlist[str(count)] = {}

            while True:

                bbox_test[0] = bbox_test[0] - int(search_step / 2)
                bbox_test[1] = bbox_test[1] - int(search_step / 2)

                bbox_test[2] = bbox_test[2] + search_step
                bbox_test[3] = bbox_test[3] + search_step

                if bbox_test[0] < 0 or bbox_test[1] < 0 or bbox_test[1] + bbox_test[3] >= height \
                        or bbox_test[0] + bbox_test[2] >= width:
                    break

                cgf = getGF_fromintegral(integral_grad, bbox_test) / (bbox_test[2] * bbox_test[3])
                if cgf < gf:
                    break
                bbox = bbox_test.copy()

            patchlist[str(count)]['rect'] = bbox
            patchlist[str(count)]['size'] = bbox[2]
            count = count + 1

    return patchlist

def getGF_fromintegral(integralimage, rect):
    # Computes the gradient density of a given patch from the gradient integral image.
    x1 = rect[1]
    x2 = rect[1] + rect[3]
    y1 = rect[0]
    y2 = rect[0] + rect[2]
    value = integralimage[x2, y2] - integralimage[x1, y2] - integralimage[x2, y1] + integralimage[x1, y1]
    return value

def estimatemidasBoost(img, model, w, h):
    # init transform
    transform = Compose(
        [
            Resize(
                w,
                h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    # transform input
    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
        prediction = model.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction
