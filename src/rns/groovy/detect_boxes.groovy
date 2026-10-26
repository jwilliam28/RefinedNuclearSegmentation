// RefinedNuclearSegmentation: detection step, run headless by QuPath 0.6+/0.7 ("QuPath script -i <image> -a ... detect_boxes.groovy").
// In square boxes of the current image it runs
//   - StarDist (fine pass: low threshold, small model pixel) and StarDist (coarse pass: default threshold and pixel),
//   - optionally InstanSeg,
// exports each pass as QuPath GeoJSON (full-resolution image coordinates) and writes each box as a PNG.
//
// args: [0] boxes CSV "x,y,box_um" (box centres in full-resolution pixels; header line starting with 'x' is skipped)
//       [1] output directory (created if missing)
//       [2] StarDist model .pb (e.g. he_heavy_augment.pb)
//       [3] InstanSeg model directory, or '-' to skip InstanSeg
//       [4] pixel size override in um (0 = use the image metadata)
//       [5] options "key=value;..." : fine_thr (0.35) fine_px (0.35) coarse_thr (0.5) coarse_px (0.5) papmap (1) regions (1)
//
// papmap=1 feeds the models (max(R,G), G, B) instead of (R, G, B). On Papanicolaou stain the cyan (light green) cytoplasm
// absorbs red but not green, so in raw RGB it is as dark in the red channel as a nucleus and H&E-trained models take it
// for nuclear material; max(R,G) makes cyan cytoplasm bright, keeps nuclei (which absorb red AND green) dark and leaves
// pink (eosin) cytoplasm pink.
import qupath.ext.stardist.StarDist2D
import qupath.lib.images.servers.ColorTransforms
import qupath.lib.io.GsonTools
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.ROIs
import qupath.opencv.ops.ImageOps

def a = args
def rows = new File(a[0]).readLines().findAll { it.trim() && !it.trim().startsWith('x') }.collect { it.split(',')*.trim()*.toDouble() }
def outDir = new File(a[1]); outDir.mkdirs()
// outputs of an earlier run in the same folder must never be mistaken for this run's
['stardist_fine.geojson', 'stardist_coarse.geojson', 'instanseg.geojson', 'boxes.json'].each { new File(outDir, it).delete() }
new File(outDir, 'regions').listFiles()?.findAll { it.name ==~ /box_\d+\.png/ }?.each { it.delete() }
def sdModel = a[2]
def isModel = a.length > 3 ? a[3] : '-'
double pxOverride = a.length > 4 ? (a[4] as double) : 0
def opt = [fine_thr: '0.35', fine_px: '0.35', coarse_thr: '0.5', coarse_px: '0.5', papmap: '1', regions: '1']
if (a.length > 5 && a[5].trim()) {
    a[5].split(';').findAll { it.contains('=') }.each { kv -> def p = kv.split('=', 2); opt[p[0].trim()] = p[1].trim() }
}
boolean papmap = opt.papmap == '1'

def imageData = getCurrentImageData()
def server = imageData.getServer()
double px = pxOverride > 0 ? pxOverride : server.getPixelCalibration().getAveragedPixelSizeMicrons()
if (pxOverride > 0 || Double.isNaN(px)) setPixelSizeMicrons(px, px)
println("RNS_PIXEL_SIZE_UM " + px + " IMAGE " + server.getWidth() + "x" + server.getHeight() + " BOXES " + rows.size())

def hier = imageData.getHierarchy()
hier.clearAll()
def annos = []
def boxes = []
rows.eachWithIndex { p, i ->
    int half = (int) Math.round(p[2] / px / 2.0)
    int x0 = Math.max(0, (int) Math.round(p[0]) - half), y0 = Math.max(0, (int) Math.round(p[1]) - half)
    int w = Math.min(2 * half, server.getWidth() - x0), h = Math.min(2 * half, server.getHeight() - y0)
    def roi = ROIs.createRectangleROI(x0, y0, w, h, ImagePlane.getDefaultPlane())
    def an = PathObjects.createAnnotationObject(roi)
    an.setName(String.format("box_%04d", i + 1))
    annos << an
    boxes << [id: i + 1, x0: x0, y0: y0, w: w, h: h, cx: p[0], cy: p[1], box_um: p[2]]
}
hier.addObjects(annos)

def collect = { String name ->
    def dets = []
    annos.eachWithIndex { an, i ->
        an.getChildObjects().each { d ->
            if (d.isDetection()) { d.getMeasurementList().put("box_id", (double) (i + 1)); dets << d }
        }
    }
    new File(outDir, name).text = GsonTools.getInstance(false).toJson(dets)
    println("RNS_WROTE " + name + " " + dets.size())
}

def runStarDist = { double thr, double sdPx, String name ->
    def b = StarDist2D.builder(sdModel).threshold(thr)
    if (papmap) {
        b = b.preprocess(ImageOps.Core.splitMerge(
                ImageOps.Core.sequential(ImageOps.Channels.extract(0, 1), ImageOps.Channels.maximum()),
                ImageOps.Channels.extract(1), ImageOps.Channels.extract(2)))
    }
    def sd = b.normalizePercentiles(1, 99).pixelSize(sdPx).measureShape().build()
    sd.detectObjects(imageData, annos)
    sd.close()
    collect(name)
}
runStarDist(opt.fine_thr as double, opt.fine_px as double, 'stardist_fine.geojson')
runStarDist(opt.coarse_thr as double, opt.coarse_px as double, 'stardist_coarse.geojson')

def isStatus = 'skipped'
new File(outDir, 'instanseg.geojson').text = '[]'
if (isModel != '-') {
    try {
        def maxRG = [
                extractChannel: { srv, img, pixels ->
                    int w = img.getWidth(), h = img.getHeight()
                    def raster = img.getRaster()
                    float[] r = raster.getSamples(0, 0, w, h, 0, (float[]) null)
                    float[] g = raster.getSamples(0, 0, w, h, 1, (float[]) null)
                    int npx = w * h
                    float[] res = new float[npx]
                    for (int i = 0; i < npx; i++) { res[i] = Math.max(r[i], g[i]) }
                    return res
                },
                getName       : { -> 'max(Red,Green)' },
                supportsImage : { srv -> true }
        ] as ColorTransforms.ColorTransform
        def chans = papmap ? [maxRG, ColorTransforms.createChannelExtractor(1), ColorTransforms.createChannelExtractor(2)]
                           : [ColorTransforms.createChannelExtractor(0), ColorTransforms.createChannelExtractor(1), ColorTransforms.createChannelExtractor(2)]
        def cls = Class.forName('qupath.ext.instanseg.core.InstanSeg')
        def seg = cls.builder().modelPath(isModel).device("cpu").nThreads(4).tileDims(512).interTilePadding(32)
                .inputChannels(chans).outputChannels().makeMeasurements(true).randomColors(false).build()
        seg.detectObjects(imageData, annos)
        collect('instanseg.geojson')
        isStatus = 'ok'
    } catch (Exception | LinkageError t) {      // missing extension, PyTorch/DJL not set up, API change; not OOM
        isStatus = 'failed'
        new File(outDir, 'instanseg.geojson').text = '[]'
        println("RNS_INSTANSEG_SKIPPED " + t.getClass().getSimpleName() + ": " + t.getMessage())
    }
}

if (opt.regions == '1') {
    def rd = new File(outDir, 'regions'); rd.mkdirs()
    boxes.each { b ->
        def req = RegionRequest.createInstance(server.getPath(), 1.0, b.x0, b.y0, b.w, b.h)
        writeImageRegion(server, req, new File(rd, String.format("box_%04d.png", b.id)).getAbsolutePath())
    }
    println("RNS_REGIONS " + boxes.size())
}
new File(outDir, 'boxes.json').text = GsonTools.getInstance(true).toJson([pixel_size_um: px, instanseg: isStatus, papmap: papmap, boxes: boxes])
println("RNS_DONE")
