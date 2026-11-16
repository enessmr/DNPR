// SPDX-License-Identifier: GPL-2.0
/* Panic-only PCM output to the connected Microstep over NVIDIA HDA.
 * All allocation, decoding, and PCI mapping happen during normal boot.
 * The panic path takes no ALSA lock, allocates nothing, and never sleeps.
 * This intentionally takes over one HDA output stream after the system has
 * panicked; ordinary ALSA ownership remains untouched until that moment.
 */
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/ktime.h>
#include <linux/pci.h>
#include <linux/printk.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/vmalloc.h>
#include <linux/zlib.h>

#include <sound/hda_register.h>
#include <sound/hda_verbs.h>

#include "dnpr_audio.h"
#include "dnpr_audio_blob.h"

#define DNPR_HDA_VENDOR 0x10de
#define DNPR_HDA_DEVICE 0x10f1
#define DNPR_HDA_PIN 0x05
#define DNPR_HDA_CONVERTER 0x0a
#define DNPR_HDA_CODEC 0
#define DNPR_HDA_FORMAT 0x0a11 /* 32 kHz, 16-bit, two channels */
#define DNPR_HDA_CHUNK 4096U
#define DNPR_HDA_ENTRIES DIV_ROUND_UP(DNPR_AUDIO_BYTES, DNPR_HDA_CHUNK)

struct dnpr_hda_bdle {
	__le32 addr_lo, addr_hi, length, flags;
};

static struct pci_dev *dnpr_hda_pci;
static void __iomem *dnpr_hda_regs;
static void *dnpr_hda_pcm[DNPR_HDA_ENTRIES];
static dma_addr_t dnpr_hda_pcm_dma[DNPR_HDA_ENTRIES];
static struct dnpr_hda_bdle *dnpr_hda_bdl;
static dma_addr_t dnpr_hda_bdl_dma;
static unsigned int dnpr_hda_stream_index;
static bool dnpr_hda_ready;
static bool dnpr_hda_started;
static u64 dnpr_hda_start_ns;

static bool dnpr_hda_verb(unsigned int nid, unsigned int verb, unsigned int value)
{
	u32 cmd = (DNPR_HDA_CODEC << 28) | (nid << 20) | (verb << 8) | value;
	unsigned int n;

	for (n = 0; n < 100; n++) {
		if (!(readw(dnpr_hda_regs + AZX_REG_IRS) & AZX_IRS_BUSY))
			break;
		udelay(1);
	}
	if (n == 100)
		return false;
	writew(AZX_IRS_VALID, dnpr_hda_regs + AZX_REG_IRS);
	writel(cmd, dnpr_hda_regs + AZX_REG_IC);
	writew(AZX_IRS_BUSY, dnpr_hda_regs + AZX_REG_IRS);
	for (n = 0; n < 1000; n++) {
		if (readw(dnpr_hda_regs + AZX_REG_IRS) & AZX_IRS_VALID)
			return true;
		udelay(1);
	}
	return false;
}

void dnpr_audio_start(void)
{
	void __iomem *sd;
	u32 ctl;
	unsigned int n;

	if (!READ_ONCE(dnpr_hda_ready) || dnpr_hda_started)
		return;
	dnpr_hda_started = true;
	sd = dnpr_hda_regs + 0x80 + 0x20 * dnpr_hda_stream_index;
	ctl = readl(sd + AZX_REG_SD_CTL) & ~SD_CTL_DMA_START;
	writel(ctl, sd + AZX_REG_SD_CTL);
	udelay(10);
	writel(ctl | SD_CTL_STREAM_RESET, sd + AZX_REG_SD_CTL);
	for (n = 0; n < 100; n++) {
		if (readl(sd + AZX_REG_SD_CTL) & SD_CTL_STREAM_RESET)
			break;
		udelay(1);
	}
	writel(ctl & ~SD_CTL_STREAM_RESET, sd + AZX_REG_SD_CTL);
	for (n = 0; n < 100; n++) {
		if (!(readl(sd + AZX_REG_SD_CTL) & SD_CTL_STREAM_RESET))
			break;
		udelay(1);
	}
	if (n == 100)
		return;

	/* A direct stream is needed here: ALSA's normal start path can sleep or
	 * wait on a lock held by a CPU already stopped by panic(). */
	writeb(0, dnpr_hda_regs + AZX_REG_CORBCTL);
	writeb(0, dnpr_hda_regs + AZX_REG_RIRBCTL);
	if (!dnpr_hda_verb(DNPR_HDA_CONVERTER, AC_VERB_SET_POWER_STATE, 0) ||
	    !dnpr_hda_verb(DNPR_HDA_PIN, AC_VERB_SET_POWER_STATE, 0) ||
	    !dnpr_hda_verb(DNPR_HDA_CONVERTER, AC_VERB_SET_CHANNEL_STREAMID, 0x10) ||
	    !dnpr_hda_verb(DNPR_HDA_CONVERTER, AC_VERB_SET_STREAM_FORMAT, DNPR_HDA_FORMAT) ||
	    !dnpr_hda_verb(DNPR_HDA_CONVERTER, AC_VERB_SET_DIGI_CONVERT_1, AC_DIG1_ENABLE) ||
	    !dnpr_hda_verb(DNPR_HDA_PIN, AC_VERB_SET_PIN_WIDGET_CONTROL, AC_PINCTL_OUT_EN))
		return;

	writel(lower_32_bits(dnpr_hda_bdl_dma), sd + AZX_REG_SD_BDLPL);
	writel(upper_32_bits(dnpr_hda_bdl_dma), sd + AZX_REG_SD_BDLPU);
	writel(DNPR_AUDIO_BYTES, sd + AZX_REG_SD_CBL);
	writew(DNPR_HDA_ENTRIES - 1, sd + AZX_REG_SD_LVI);
	writew(DNPR_HDA_FORMAT, sd + AZX_REG_SD_FORMAT);
	writel((1U << SD_CTL_STREAM_TAG_SHIFT) | SD_CTL_DMA_START,
	       sd + AZX_REG_SD_CTL);
	dnpr_hda_start_ns = ktime_get_mono_fast_ns();
}

void dnpr_audio_maybe_stop(void)
{
	void __iomem *sd;

	if (!dnpr_hda_started || !dnpr_hda_regs || !dnpr_hda_start_ns ||
	    ktime_get_mono_fast_ns() - dnpr_hda_start_ns < 6000000000ULL)
		return;
	sd = dnpr_hda_regs + 0x80 + 0x20 * dnpr_hda_stream_index;
	writel(readl(sd + AZX_REG_SD_CTL) & ~SD_CTL_DMA_START,
	       sd + AZX_REG_SD_CTL);
	dnpr_hda_started = false;
}

static int __init dnpr_hda_init(void)
{
	u8 *decoded = NULL;
	u16 gcap;
	unsigned int i;

	dnpr_hda_pci = pci_get_device(DNPR_HDA_VENDOR, DNPR_HDA_DEVICE, NULL);
	if (!dnpr_hda_pci)
		return 0;
	dnpr_hda_regs = pci_iomap(dnpr_hda_pci, 0, 0);
	if (!dnpr_hda_regs) {
		pr_warn("DNPR: HDA BAR 0 map failed\n");
		goto fail;
	}
	gcap = readw(dnpr_hda_regs + AZX_REG_GCAP);
	dnpr_hda_stream_index = (gcap & AZX_GCAP_ISS) >> 8;
	if (!((gcap & AZX_GCAP_OSS) >> 12) ||
	    dnpr_hda_stream_index > 14 || DNPR_HDA_ENTRIES > 256) {
		pr_warn("DNPR: HDA GCAP %#x not usable for panic PCM\n", gcap);
		goto fail;
	}
	dnpr_hda_bdl = dma_alloc_coherent(&dnpr_hda_pci->dev,
					 sizeof(*dnpr_hda_bdl) * DNPR_HDA_ENTRIES,
					 &dnpr_hda_bdl_dma, GFP_KERNEL);
	decoded = vmalloc(DNPR_AUDIO_BYTES);
	if (!dnpr_hda_bdl || !decoded) {
		pr_warn("DNPR: HDA PCM allocation failed\n");
		goto fail;
	}
	if (zlib_inflate_blob(decoded, DNPR_AUDIO_BYTES,
			      dnpr_audio_deflate,
			      sizeof(dnpr_audio_deflate)) != DNPR_AUDIO_BYTES) {
		pr_warn("DNPR: HDA PCM decode failed\n");
		goto fail;
	}
	for (i = 0; i < DNPR_HDA_ENTRIES; i++) {
		unsigned int length = min_t(unsigned int, DNPR_HDA_CHUNK,
						   DNPR_AUDIO_BYTES - i * DNPR_HDA_CHUNK);
		dnpr_hda_pcm[i] = dma_alloc_coherent(&dnpr_hda_pci->dev,
						    DNPR_HDA_CHUNK,
						    &dnpr_hda_pcm_dma[i], GFP_KERNEL);
		if (!dnpr_hda_pcm[i])
			goto fail;
		memcpy(dnpr_hda_pcm[i], decoded + i * DNPR_HDA_CHUNK, length);
		dnpr_hda_bdl[i].addr_lo = cpu_to_le32(lower_32_bits(dnpr_hda_pcm_dma[i]));
		dnpr_hda_bdl[i].addr_hi = cpu_to_le32(upper_32_bits(dnpr_hda_pcm_dma[i]));
		dnpr_hda_bdl[i].length = cpu_to_le32(length);
		dnpr_hda_bdl[i].flags = 0;
	}
	vfree(decoded);
	WRITE_ONCE(dnpr_hda_ready, true);
	pr_info("DNPR: panic HDA PCM prepared for Microstep HDMI (32 kHz stereo)\n");
	return 0;
fail:
	vfree(decoded);
	for (i = 0; i < DNPR_HDA_ENTRIES; i++)
		if (dnpr_hda_pcm[i])
			dma_free_coherent(&dnpr_hda_pci->dev, DNPR_HDA_CHUNK,
					  dnpr_hda_pcm[i], dnpr_hda_pcm_dma[i]);
	if (dnpr_hda_bdl)
		dma_free_coherent(&dnpr_hda_pci->dev,
				  sizeof(*dnpr_hda_bdl) * DNPR_HDA_ENTRIES,
				  dnpr_hda_bdl, dnpr_hda_bdl_dma);
	if (dnpr_hda_regs)
		pci_iounmap(dnpr_hda_pci, dnpr_hda_regs);
	pci_dev_put(dnpr_hda_pci);
	pr_warn("DNPR: panic HDA PCM unavailable\n");
	return 0;
}
late_initcall(dnpr_hda_init);
