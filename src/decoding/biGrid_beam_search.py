import torch

from src.data.vocabulary import BOS, EOS, PAD
from src.models.base import NMTModel
from .utils import mask_scores, tensor_gather_helper


def biGrid_beam_search(fw_nmt_model, bw_nmt_model, beam_size, max_steps, src_seqs, alpha, constrains, imt_step=1):
    batch_size = len(constrains)

    patience = len(constrains[0])

    fw_enc_outputs = fw_nmt_model.encode(src_seqs)
    bw_enc_outputs = bw_nmt_model.encode(src_seqs)

    fw_init_dec_states = fw_nmt_model.init_decoder(fw_enc_outputs, expand_size=beam_size)
    bw_init_dec_states = bw_nmt_model.init_decoder(bw_enc_outputs, expand_size=beam_size)

    total_scores = None
    total_ids = None
    total_lengths = None

    for p in range(patience):
        beam_mask = src_seqs.new(batch_size, beam_size).fill_(1).float()
        final_lengths = src_seqs.new(batch_size, beam_size).zero_().float()
        beam_scores = src_seqs.new(batch_size, beam_size).zero_().float()
        final_word_indices = src_seqs.new(batch_size, beam_size, 1).fill_(BOS)

        grid_word_indices = [final_word_indices.clone() for i in range(imt_step + 1)]
        grid_beam_scores = [beam_scores.clone() for i in range(imt_step + 1)]
        grid_beam_mask = [beam_mask.clone() for i in range(imt_step + 1)]
        grid_final_lengths = [final_lengths.clone() for i in range(imt_step + 1)]

        grid_dec_states = [fw_nmt_model.copy_dec_states(fw_init_dec_states, beam_size) for i in range(imt_step + 1)]

        for t in range(max_steps):
            grid_word_buf = [[] for i in range(imt_step + 1)]
            grid_score_buf = [[] for i in range(imt_step + 1)]
            grid_mask_buf = [[] for i in range(imt_step + 1)]
            grid_lengths_buf = [[] for i in range(imt_step + 1)]
            grid_state_buf = [[] for i in range(imt_step + 1)]

            to_be_flag = False

            for c in range(imt_step + 1):
                if t < c:
                    break
                fw_dec_states = grid_dec_states[c]
                final_word_indices = grid_word_indices[c].clone()
                beam_scores = grid_beam_scores[c].clone()
                beam_mask = grid_beam_mask[c].clone()
                if beam_mask.eq(0.0).all():
                    break
                to_be_flag = True
                final_lengths = grid_final_lengths[c].clone()

                next_scores, fw_dec_states = fw_nmt_model.decode(final_word_indices.view(batch_size * beam_size, -1),
                                                                 fw_dec_states)
                next_scores = -next_scores
                next_scores = next_scores.view(batch_size, beam_size, -1)
                next_scores = mask_scores(scores=next_scores, beam_mask=beam_mask)

                beam_scores = next_scores + beam_scores.unsqueeze(2)
                vocab_size = beam_scores.size(-1)

                if t == 0 and beam_size > 1:
                    beam_scores[:, 1:, :] = float('inf')

                if alpha > 0.0:
                    normed_scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + beam_mask + final_lengths).unsqueeze(
                        2) ** alpha
                else:
                    normed_scores = beam_scores.detach().clone()

                # normed_scores = normed_scores.view(batch_size, -1)

                # no constrains
                NC_normed_scores = normed_scores.clone()
                for b in range(batch_size):
                    cons = constrains[b][p]
                    NC_normed_scores[b, :, cons] = float('inf')

                NC_normed_scores = NC_normed_scores.view(batch_size, -1)

                _, indices = torch.topk(NC_normed_scores, k=beam_size, dim=-1, largest=False, sorted=True)
                next_beam_ids = torch.div(indices, vocab_size)
                next_word_ids = indices % vocab_size

                nc_beam_scores = beam_scores.clone()
                nc_beam_scores = nc_beam_scores.view(batch_size, -1)
                nc_beam_scores = torch.gather(nc_beam_scores, 1, indices)

                nc_beam_mask = beam_mask.clone()
                nc_beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                                    gather_from=nc_beam_mask,
                                                    batch_size=batch_size,
                                                    beam_size=beam_size,
                                                    gather_shape=[-1])

                nc_final_word_indices = final_word_indices.clone()
                nc_final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                             gather_from=nc_final_word_indices,
                                                             batch_size=batch_size,
                                                             beam_size=beam_size,
                                                             gather_shape=[batch_size * beam_size, -1])

                nc_final_lengths = final_lengths.clone()
                nc_final_lengths = tensor_gather_helper(gather_indices=next_beam_ids,
                                                        gather_from=nc_final_lengths,
                                                        batch_size=batch_size,
                                                        beam_size=beam_size,
                                                        gather_shape=[-1])

                nc_dec_state = fw_nmt_model.copy_dec_states(fw_dec_states, beam_size)
                nc_dec_state = fw_nmt_model.reorder_dec_states(nc_dec_state, new_beam_indices=next_beam_ids,
                                                               beam_size=beam_size)

                nc_beam_mask_ = 1.0 - next_word_ids.eq(EOS).float()
                next_word_ids.masked_fill_((nc_beam_mask_ + nc_beam_mask).eq(0.0), PAD)
                nc_beam_mask = nc_beam_mask * nc_beam_mask_

                nc_final_lengths += nc_beam_mask

                nc_final_word_indices = torch.cat((nc_final_word_indices, next_word_ids.unsqueeze(2)), dim=2)
                if c == 0:
                    grid_word_buf[c] = nc_final_word_indices
                    grid_score_buf[c] = nc_beam_scores
                    grid_mask_buf[c] = nc_beam_mask
                    grid_lengths_buf[c] = nc_final_lengths
                    grid_state_buf[c] = fw_nmt_model.copy_dec_states(nc_dec_state, beam_size)
                elif grid_score_buf[c][0][0] > nc_beam_scores[0][0]:
                    grid_word_buf[c] = nc_final_word_indices
                    grid_score_buf[c] = nc_beam_scores
                    grid_mask_buf[c] = nc_beam_mask
                    grid_lengths_buf[c] = nc_final_lengths
                    grid_state_buf[c] = fw_nmt_model.copy_dec_states(nc_dec_state, beam_size)

                # with constrains
                if c < imt_step:
                    WC_normed_scores = normed_scores.clone()
                    for b in range(batch_size):
                        cons = constrains[b][p]
                        WC_normed_scores[b, :, cons] = float('-inf')

                    WC_normed_scores = WC_normed_scores.view(batch_size, -1)

                    _, indices = torch.topk(WC_normed_scores, k=beam_size, dim=-1, largest=False, sorted=True)
                    next_beam_ids = torch.div(indices, vocab_size)
                    next_word_ids = indices % vocab_size

                    wc_beam_scores = beam_scores.clone()
                    wc_beam_scores = wc_beam_scores.view(batch_size, -1)
                    wc_beam_scores = torch.gather(wc_beam_scores, 1, indices)

                    wc_beam_mask = beam_mask.clone()
                    wc_beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                                        gather_from=wc_beam_mask,
                                                        batch_size=batch_size,
                                                        beam_size=beam_size,
                                                        gather_shape=[-1])

                    wc_final_word_indices = final_word_indices.clone()
                    wc_final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                                 gather_from=wc_final_word_indices,
                                                                 batch_size=batch_size,
                                                                 beam_size=beam_size,
                                                                 gather_shape=[batch_size * beam_size, -1])

                    wc_final_lengths = final_lengths.clone()
                    wc_final_lengths = tensor_gather_helper(gather_indices=next_beam_ids,
                                                            gather_from=wc_final_lengths,
                                                            batch_size=batch_size,
                                                            beam_size=beam_size,
                                                            gather_shape=[-1])

                    wc_dec_state = fw_nmt_model.copy_dec_states(fw_dec_states, beam_size)
                    wc_dec_state = fw_nmt_model.reorder_dec_states(wc_dec_state, new_beam_indices=next_beam_ids,
                                                                   beam_size=beam_size)

                    wc_beam_mask_ = 1.0 - next_word_ids.eq(EOS).float()
                    next_word_ids.masked_fill_((wc_beam_mask_ + wc_beam_mask).eq(0.0), PAD)
                    wc_beam_mask = wc_beam_mask * wc_beam_mask_

                    wc_final_lengths += wc_beam_mask

                    wc_final_word_indices = torch.cat((wc_final_word_indices, next_word_ids.unsqueeze(2)), dim=2)

                    grid_word_buf[c + 1] = wc_final_word_indices
                    grid_score_buf[c + 1] = wc_beam_scores
                    grid_mask_buf[c + 1] = wc_beam_mask
                    grid_lengths_buf[c + 1] = wc_final_lengths
                    grid_state_buf[c + 1] = fw_nmt_model.copy_dec_states(wc_dec_state, beam_size)

            if to_be_flag == False:
                break
            grid_word_indices = grid_word_buf
            grid_beam_scores = grid_score_buf
            grid_beam_mask = grid_mask_buf
            grid_final_lengths = grid_lengths_buf

            grid_dec_states = grid_state_buf

        fixed_word_ids = grid_word_indices[imt_step].clone()
        fixed_word_ids = fixed_word_ids[:, :, 1:]
        for ia in range(batch_size):
            for ib in range(beam_size):
                ic = 0
                while ic < len(fixed_word_ids[ia, ib]) and fixed_word_ids[ia, ib, ic] != EOS:
                    ic += 1
                ic -= 1
                for id in range(int(int(ic) / 2) + 1):
                    tmp = int(fixed_word_ids[ia, ib, id])
                    fixed_word_ids[ia, ib, id] = int(fixed_word_ids[ia, ib, ic - id])
                    fixed_word_ids[ia, ib, ic - id] = tmp

        # backward
        beam_mask = src_seqs.new(batch_size, beam_size).fill_(1).float()
        final_lengths = src_seqs.new(batch_size, beam_size).zero_().float()
        beam_scores = src_seqs.new(batch_size, beam_size).zero_().float()
        final_word_indices = src_seqs.new(batch_size, beam_size, 1).fill_(BOS)
        dec_states = bw_nmt_model.copy_dec_states(bw_init_dec_states, beam_size)

        swicth = [[True for iii in range(beam_size)] for jjj in range(batch_size)]

        for t in range(max_steps):
            next_scores, dec_states = bw_nmt_model.decode(final_word_indices.view(batch_size * beam_size, -1),
                                                          dec_states)

            next_scores = - next_scores  # convert to negative log_probs

            next_scores = next_scores.view(batch_size, beam_size, -1)
            next_scores = mask_scores(scores=next_scores, beam_mask=beam_mask)

            beam_scores = next_scores + beam_scores.unsqueeze(2)  # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]

            vocab_size = beam_scores.size(-1)
            if t == 0 and beam_size > 1:
                # Force to select first beam at step 0
                beam_scores[:, 1:, :] = float('inf')

                # Length penalty
            if alpha > 0.0:
                normed_scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + beam_mask + final_lengths).unsqueeze(
                    2) ** alpha
            else:
                normed_scores = beam_scores.detach().clone()
            if t < len(fixed_word_ids[0][0]):
                for b in range(batch_size):
                    cons = constrains[b][p]
                    for bb in range(beam_size):
                        w = fixed_word_ids[b, bb, t]
                        if swicth[b][bb] == True:
                            normed_scores[b, bb, w] = float('-inf')
                        if w == cons:
                            swicth[b][bb] = False

            normed_scores = normed_scores.view(batch_size, -1)

            # Get topK with beams
            # indices: [batch_size, ]
            _, indices = torch.topk(normed_scores, k=beam_size, dim=-1, largest=False, sorted=True)
            next_beam_ids = torch.div(indices, vocab_size)  # [batch_size, ]
            next_word_ids = indices % vocab_size  # [batch_size, ]

            # Re-arrange by new beam indices
            beam_scores = beam_scores.view(batch_size, -1)
            beam_scores = torch.gather(beam_scores, 1, indices)

            beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                             gather_from=beam_mask,
                                             batch_size=batch_size,
                                             beam_size=beam_size,
                                             gather_shape=[-1])

            final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                      gather_from=final_word_indices,
                                                      batch_size=batch_size,
                                                      beam_size=beam_size,
                                                      gather_shape=[batch_size * beam_size, -1])

            final_lengths = tensor_gather_helper(gather_indices=next_beam_ids,
                                                 gather_from=final_lengths,
                                                 batch_size=batch_size,
                                                 beam_size=beam_size,
                                                 gather_shape=[-1])

            dec_states = bw_nmt_model.reorder_dec_states(dec_states, new_beam_indices=next_beam_ids,
                                                         beam_size=beam_size)

            # If next_word_ids is EOS, beam_mask_ should be 0.0
            beam_mask_ = 1.0 - next_word_ids.eq(EOS).float()
            next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0),
                                       PAD)  # If last step a EOS is already generated, we replace the last token as PAD
            beam_mask = beam_mask * beam_mask_

            # # If an EOS or PAD is encountered, set the beam mask to 0.0
            final_lengths += beam_mask

            final_word_indices = torch.cat((final_word_indices, next_word_ids.unsqueeze(2)), dim=2)

            if beam_mask.eq(0.0).all():
                break
        final_word_indices = final_word_indices[:, :, 1:].contiguous()
        for ia in range(batch_size):
            for ib in range(beam_size):
                ic = 0
                while ic < len(final_word_indices[ia, ib]) and final_word_indices[ia, ib, ic] != EOS:
                    ic += 1
                ic -= 1
                for id in range(int(int(ic) / 2) + 1):
                    tmp = int(final_word_indices[ia, ib, id])
                    final_word_indices[ia, ib, id] = int(final_word_indices[ia, ib, ic - id])
                    final_word_indices[ia, ib, ic - id] = tmp

        if alpha > 0.0:
            scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + final_lengths) ** alpha
        else:
            scores = beam_scores / final_lengths

        _, reranked_ids = torch.sort(scores, dim=-1, descending=False)
        final_word_indices = tensor_gather_helper(gather_indices=reranked_ids,
                             gather_from=final_word_indices,
                             batch_size=batch_size,
                             beam_size=beam_size,
                             gather_shape=[batch_size * beam_size, -1])
        if p == 0:
            total_scores = beam_scores.clone()
            total_ids = final_word_indices.clone()
            total_lengths = final_lengths.clone()
        else:
            total_scores = torch.cat((total_scores, beam_scores), dim=1)
            t_size = total_ids.size(-1)
            f_size = final_word_indices.size(-1)
            if t_size < f_size:
                pad_ids = src_seqs.new(batch_size, total_ids.size(1), f_size - t_size).fill_(PAD)
                total_ids = torch.cat((total_ids, pad_ids), dim=2)
            elif t_size > f_size:
                pad_ids = src_seqs.new(batch_size, beam_size, t_size - f_size).fill_(PAD)
                final_word_indices = torch.cat((final_word_indices, pad_ids), dim=2)
            total_ids = torch.cat((total_ids, final_word_indices), dim=1)
            total_lengths = torch.cat((total_lengths, final_lengths), dim=1)



    return total_ids
