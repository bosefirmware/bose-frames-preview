var Bose=Bose||{};Bose.productImageSlider2=Bose.productImageSlider2||{},Bose.productImageSlider2.selectors={component:".bose-productImageSlider2",slider:".bose-productImageSlider2Container",sliderID:"#bose-productImageSliderContainer",slide:"div .bose-productImageSlider2Container .slick-slide",sizeVariant:".bose-variantBlockSelector__block",productStatus:'input[name="productStatus-variantCode"]',productVariant:'input[name=":product-variantCode"]',productPath:'input[name=":product-path"]',productStatusContainer:".bose-productStatus__container",sliderLast:".bose-productImageSlider2Container:last",slickSlides:".bose-productImageSlider2 .slider-nav .slick-slide",sliderDotsList:".slick-dots li",activeSliderDot:".slick-active",staticContainer:".bose-staticContainer",navigationSlider:".bose-productImageSlider2 .slider-nav",fullscreenIcon:".bose-fullscreenIcon",ecommerce2mediaArea:".js-ecommerceArea2-mediaArea",closeZoom:".bose-productImageSlider2_closeZoom",showFullscreen:".bose-productImageSlider2__fullscreen",closeFullscreen:".bose-productImageSlider2__closescreen",animateZoomIn:".bose-animate__productImageSlider2",animateZoomOut:".bose-animate__productImageSlider2-zoomOut",slickTrack:".bose-productImageSlider2Container .slick-track"},Bose.productImageSlider2.options={history:!1,focus:!1,shareEl:!1,showAnimationDuration:0,hideAnimationDuration:0,closeEl:!0,fullscreenEl:!1,zoomEl:!1,counterEl:!0,arrowEl:!0,count:0,index:0,widthFactor:0,slickWidth:0},Bose.productImageSlider2.size={w:500,h:400},Bose.productImageSlider2.init=function(){var a=$jq(Bose.productImageSlider2.selectors.sliderLast);Bose.productImageSlider2.$slider=a,Bose.productImageSlider2.$slider.length>0&&(Bose.productImageSlider2.defaultVariant=Bose.productImageSlider2.$slider.data("default-variant-code"),Bose.productImageSlider2.$slider.slick({lazyLoad:"ondemand",slidesToShow:1,slidesToScroll:1,arrows:!1,dots:!1,respondTo:"slider",prevArrow:'<button type="button" class="slick-prev" aria-label="'+Bose.Translation.getSiteTranslation("PREVIOUS_IMAGE")+'"></button>',nextArrow:'<button type="button" class="slick-next" aria-label="'+Bose.Translation.getSiteTranslation("NEXT_IMAGE")+'"></button>',cssEase:"linear",asNavFor:Bose.productImageSlider2.selectors.navigationSlider}),Bose.productImageSlider2.options.count=parseInt($jq(Bose.productImageSlider2.selectors.staticContainer+" ."+Bose.productImageSlider2.defaultVariant).length,10),$jq(Bose.productImageSlider2.selectors.navigationSlider).slick({slidesToShow:Bose.productImageSlider2.options.count-1,slidesToScroll:1,asNavFor:Bose.productImageSlider2.selectors.sliderLast,dots:!1,centerMode:!1}),$jq(document).on("click",Bose.productImageSlider2.selectors.slickSlides,Bose.productImageSlider2.showSelectedSlide).on("click",Bose.productImageSlider2.selectors.showFullscreen,Bose.productImageSlider2.openZoom).on("click",Bose.productImageSlider2.selectors.closeFullscreen,Bose.productImageSlider2.closeZoom).on("click",Bose.productImageSlider2.selectors.closeZoom,Bose.productImageSlider2.closeZoom).on("transitionend",Bose.productImageSlider2.selectors.animateZoomIn,Bose.productImageSlider2.zoomedIn).on("transitionend",Bose.productImageSlider2.selectors.animateZoomOut,Bose.productImageSlider2.zoomedOut),Bose.productImageSlider2.$slider.slick("slickFilter","."+Bose.productImageSlider2.defaultVariant),$jq(Bose.productImageSlider2.selectors.navigationSlider).slick("slickFilter","."+Bose.productImageSlider2.defaultVariant),Bose.productImageSlider2.unHide(),$jq(document).on("variantChanged incentiveVariantChanged",Bose.productImageSlider2.switchVariant),Bose.productImageSlider2.zoom(),Bose.productImageSlider2.setOgImage(),Gridle.isActive("large")?Bose.productImageSlider2.options.widthFactor=10.22:Gridle.isActive("medium")?Bose.productImageSlider2.options.widthFactor=13.1:Gridle.isActive("small")&&($jq(Bose.productImageSlider2.selectors.fullscreenIcon).addClass("hide"),Bose.productImageSlider2.options.widthFactor=16),Bose.productImageSlider2.setCount(Bose.productImageSlider2.options.count),Bose.productImageSlider2.$slider.on("afterChange",Bose.productImageSlider2.zoom))},Bose.productImageSlider2.zoom=function(){Gridle.isActive("small")&&Bose.productImageSlider2.initPhotoSwipeFromDOM(Bose.productImageSlider2.selectors.sliderID)},Bose.productImageSlider2.showSelectedSlide=function(){var a=$jq(this).data("slick-index");a>=Bose.productImageSlider2.options.index?Bose.productImageSlider2.$slider.slick("slickGoTo",parseInt(a,10)-Bose.productImageSlider2.options.index):Bose.productImageSlider2.$slider.slick("slickGoTo",parseInt(a,10))},Bose.productImageSlider2.openZoom=function(){var a=(Bose.productImageSlider2.options.count-1)*Bose.productImageSlider2.options.widthFactor/2;$jq(Bose.productImageSlider2.selectors.fullscreenIcon).removeClass("bose-productImageSlider2__fullscreen").addClass("bose-productImageSlider2__closescreen"),$jq(Bose.productImageSlider2.selectors.slider).removeClass("bose-productImageSlider2__zoomOut").addClass("bose-productImageSlider2__zoom"),$jq(Bose.productImageSlider2.selectors.slide+" img").addClass("bose-productImageSlider2__zoomImage"),$jq(Bose.productImageSlider2.selectors.slider).parents(".bose-ecommerceArea2__container").addClass("bose-ecommerceArea2__container-zoom"),$jq(Bose.productImageSlider2.selectors.ecommerce2mediaArea).removeClass("grid-6 grid-medium-6 bose-ecommerceArea2__mediaArea").addClass("grid-12 bose-animate__productImageSlider2 grid-medium-12"),$jq(".bose-ecommerceArea2__ctaArea, .bose-productImageSlider2Container .slick-track, .js-ecommerceArea2-mediaArea .bose-productStatusAlign").css("display","none"),$jq(Bose.productImageSlider2.selectors.closeZoom).removeClass("hide"),$jq(Bose.productImageSlider2.selectors.navigationSlider).css("width",a+"%"),Gridle.isActive("large")&&Bose.productImageSlider2.$slider.slick("slickNext")},Bose.productImageSlider2.zoomedIn=function(){var a=$jq(Bose.productImageSlider2.selectors.navigationSlider),b=a.slick("slickCurrentSlide");$jq(Bose.productImageSlider2.selectors.ecommerce2mediaArea).removeClass("bose-animate__productImageSlider2"),Bose.productImageSlider2.$slider.slick("slickSetOption","speed",500,!0),$jq(Bose.productImageSlider2.selectors.slickTrack).css("display","block"),Gridle.isActive("large")?Bose.productImageSlider2.$slider.slick("slickPrev"):a.slick("slickGoTo",b)},Bose.productImageSlider2.closeZoom=function(){var a=(Bose.productImageSlider2.options.count-1)*Bose.productImageSlider2.options.widthFactor,b=$jq(Bose.productImageSlider2.selectors.navigationSlider),c=b.slick("slickCurrentSlide");Bose.productImageSlider2.$slider.slick("slickSetOption","speed",300,!0),$jq(Bose.productImageSlider2.selectors.fullscreenIcon).removeClass("bose-productImageSlider2__closescreen").addClass("bose-productImageSlider2__fullscreen"),$jq(Bose.productImageSlider2.selectors.slider).removeClass("bose-productImageSlider2__zoom"),$jq(Bose.productImageSlider2.selectors.slide+" img").removeClass("bose-productImageSlider2__zoomImage"),$jq(".bose-ecommerceArea2__ctaArea, .js-ecommerceArea2-mediaArea .bose-productStatusAlign").css("display","block"),$jq(Bose.productImageSlider2.selectors.slider).parents(".bose-ecommerceArea2__container").removeClass("bose-ecommerceArea2__container-zoom"),$jq(Bose.productImageSlider2.selectors.slider).css("opacity","0"),$jq(Bose.productImageSlider2.selectors.ecommerce2mediaArea).removeClass("grid-12 grid-medium-12 bose-animate__productImageSlider2").addClass("grid-6 grid-medium-6 bose-ecommerceArea2__mediaArea bose-animate__productImageSlider2-zoomOut"),$jq(Bose.productImageSlider2.selectors.closeZoom).addClass("hide"),b.css("width",a+"%"),b.slick("slickGoTo",c)},Bose.productImageSlider2.zoomedOut=function(){$jq(Bose.productImageSlider2.selectors.slider).addClass("bose-productImageSlider2__zoomOut"),$jq(Bose.productImageSlider2.selectors.ecommerce2mediaArea).removeClass("bose-animate__productImageSlider2-zoomOut"),2===Bose.productImageSlider2.options.count&&$jq(Bose.productImageSlider2.selectors.slide).css("width",Bose.productImageSlider2.options.slickWidth)},Bose.productImageSlider2.switchVariant=function(a,b,c,d,e,f,g,h,i,j){if(!i){if(Bose.productImageSlider2.productCode=$jq(Bose.productImageSlider2.selectors.sliderLast).data("product-code"),Bose.productImageSlider2.productCode===b){var k=$jq(Bose.productImageSlider2.selectors.slide).length;void 0!==c&&(Bose.productImageSlider2.$slider.slick("slickUnfilter"),$jq(Bose.productImageSlider2.selectors.navigationSlider).slick("slickUnfilter"),Bose.productImageSlider2.$slider.slick("slickFilter","."+c),$jq(Bose.productImageSlider2.selectors.navigationSlider).slick("slickFilter","."+c),Bose.productImageSlider2.options.count=parseInt($jq(Bose.productImageSlider2.selectors.staticContainer+" ."+c).length,10),Bose.productImageSlider2.options.index=parseInt($jq(Bose.productImageSlider2.selectors.staticContainer+" ."+c+":first").index(),10)),0===k&&void 0!==j&&(Bose.productImageSlider2.$slider.slick("slickUnfilter"),$jq(Bose.productImageSlider2.selectors.navigationSlider).slick("slickUnfilter"),Bose.productImageSlider2.$slider.slick("slickFilter","."+j),$jq(Bose.productImageSlider2.selectors.navigationSlider).slick("slickFilter","."+j),Bose.productImageSlider2.options.count=parseInt($jq(Bose.productImageSlider2.selectors.staticContainer+" ."+j).length,10),Bose.productImageSlider2.options.index=parseInt($jq(Bose.productImageSlider2.selectors.staticContainer+" ."+j+":first").index(),10)),Bose.productImageSlider2.$slider.slick("slickCurrentSlide")>=k&&Bose.productImageSlider2.$slider.slick("slickGoTo",k-1),Bose.productImageSlider2.setCount(Bose.productImageSlider2.options.count)}lazySizes.autoSizer.checkElems(),Bose.productImageSlider2.setOgImage(),Bose.productImageSlider2.zoom(),Bose.productImageSlider2.productStatus(b)}},Bose.productImageSlider2.setCount=function(a){1===a||2===a?(Bose.productImageSlider2.options.count=2,Bose.productImageSlider2.options.slickWidth=$jq(Bose.productImageSlider2.selectors.slide).css("width")):a>8&&Gridle.isActive("large")?Bose.productImageSlider2.options.count=8:a>5&&(Gridle.isActive("small")||Gridle.isActive("medium"))&&(Bose.productImageSlider2.options.count=5);var b=(Bose.productImageSlider2.options.count-1)*Bose.productImageSlider2.options.widthFactor;$jq(Bose.productImageSlider2.selectors.navigationSlider).css("width",b+"%"),$jq(Bose.productImageSlider2.selectors.navigationSlider).slick("slickSetOption","slidesToShow",Bose.productImageSlider2.options.count-1,!0)},Bose.productImageSlider2.unHide=function(){$jq(Bose.productImageSlider2.selectors.component).removeClass("hide")},Bose.productImageSlider2.productStatus=function(a){var b=$jq("div[data-product-code="+a+"]"),c=b.find(Bose.productImageSlider2.selectors.productStatus).val();b.find(Bose.productImageSlider2.selectors.productStatusContainer).html(c)},Bose.productImageSlider2.setOgImage=function(){if($jq('meta[property="og:image"]').length&&$jq(".bose-productImageSlider2").length){var a=$jq(".bose-productImageSlider2Container img:first").attr("src"),b=window.location.origin+a;$jq('meta[property="og:image"]').attr("content",b)}},Bose.productImageSlider2.parseThumbnailElements=function(a){var b,c=[];return $jq(a).find(".slick-slide").each(function(){b={src:$jq(this).children("img").attr("data-zoom-image"),w:Bose.productImageSlider2.size.w,h:Bose.productImageSlider2.size.h},b.el=$jq(this),c.push(b)}),c},Bose.productImageSlider2.openPhotoSwipe=function(a,b){var c,d=$jq(".pswp")[0],e=Bose.productImageSlider2.parseThumbnailElements(b);Bose.productImageSlider2.options.index=parseInt(a,10),c=new PhotoSwipe(d,PhotoSwipeUI_Default,e,Bose.productImageSlider2.options),c.init()},Bose.productImageSlider2.onThumbnailsClick=function(){return Bose.productImageSlider2.openPhotoSwipe($jq(this).parent(".slick-slide").index(),$jq(this).parent(".slick-slide").closest(".slick-track")),!1},Bose.productImageSlider2.initPhotoSwipeFromDOM=function(a){$jq(a).find("img").unbind("click").bind("click",Bose.productImageSlider2.onThumbnailsClick)};
$jq(document).ready(function(){Bose.productImageSlider2.init()});
